"""Transit / exchange ring between HBM slots and pinned host DDR.

When the HBM slot pool must displace a resident expert, the victim's bytes are
staged into this bounded ring of pinned host slots *before* the incoming
expert overwrites the HBM slot (bi-directional exchange: victim HBM -> DDR on
one side, promotee DDR -> HBM on the other). The ring reuses its host slots
round-robin, so no host memory is ever (re)allocated at runtime: the arena,
its per-slot views and the ring state are built once at construction.

Zero-allocation contract: ``stage_eviction``/``promote_oldest`` perform only
in-place ``copy_`` into pre-sliced regions plus integer ring arithmetic; the
only failure mode is a full ring (:class:`ExchangeBufferFullError`), which
callers treat as back-pressure rather than growing the buffer.
"""

from __future__ import annotations

import torch

from ..core.layout import ExpertTensorLayout
from .runtime import DeviceRuntime

RING_STATE_FREE = 0
RING_STATE_DIRTY = 1  # holds staged bytes not yet promoted/consumed


class ExchangeBufferFullError(RuntimeError):
    """Raised when every transit slot is occupied (back-pressure, not growth)."""


class TransitExchangeBuffer:
    """Bounded FIFO ring of pinned host slots for HBM <-> DDR expert exchange.

    Evictions push at the head (``stage_eviction``); promotions pop from the
    tail (``promote_oldest``), keeping evicted bytes resident in DDR for at
    least a full ring's worth of subsequent traffic.
    """

    def __init__(self, runtime: DeviceRuntime, layout: ExpertTensorLayout, host_slots: int, pin: bool | None = None):
        if host_slots < 1:
            raise ValueError(f"host_slots must be >= 1, got {host_slots}")
        self._runtime = runtime
        self._layout = layout
        self._capacity = host_slots
        self._dma_stream = runtime.make_stream()
        use_pin = runtime.supports_pinned_host_memory if pin is None else pin
        if use_pin and not runtime.supports_pinned_host_memory:
            raise ValueError("pinned host memory requested but unsupported by this runtime")
        self.pinned = use_pin
        arena_flags: dict[str, object] = {"pin_memory": True} if use_pin else {}
        self._arena = torch.empty(layout.slot_num_bytes * host_slots, dtype=torch.uint8, **arena_flags)
        self._slot_regions = [
            self._arena.narrow(0, slot * layout.slot_num_bytes, layout.slot_num_bytes) for slot in range(host_slots)
        ]
        # AOT ring state: allocation-free bookkeeping at runtime.
        self._states = torch.zeros(host_slots, dtype=torch.int8)
        self._head = 0  # next eviction write position
        self._count = 0  # occupied transit slots

    # ---------------------------------------------------------------- views

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def in_flight(self) -> int:
        return self._count

    @property
    def dma_stream(self) -> object | None:
        return self._dma_stream

    def state_of(self, host_slot: int) -> int:
        return int(self._states[host_slot])

    def host_region(self, host_slot: int) -> torch.Tensor:
        """Pre-sliced pinned region of one transit slot (fixed address)."""
        return self._slot_regions[host_slot]

    def fingerprint(self) -> list[int]:
        """data_ptr of the arena and every slot region (host AOT invariant)."""
        return [self._arena.data_ptr()] + [region.data_ptr() for region in self._slot_regions]

    # ------------------------------------------------------------- transfers

    def stage_eviction(self, source_region: torch.Tensor) -> int:
        """HBM -> DDR: stage one evicted expert's slot region; returns host slot."""
        if source_region.numel() != self._layout.slot_num_bytes:
            raise ValueError(
                f"staged region holds {source_region.numel()} bytes, transit expects {self._layout.slot_num_bytes}"
            )
        if self._count == self._capacity:
            raise ExchangeBufferFullError(
                f"transit ring exhausted ({self._capacity} slots in flight); "
                "promote or consume before staging further evictions"
            )
        host_slot = (self._head + self._count) % self._capacity
        self._count += 1
        with self._runtime.stream_context(self._dma_stream):
            self._slot_regions[host_slot].copy_(source_region, non_blocking=True)
        self._states[host_slot] = RING_STATE_DIRTY
        return host_slot

    def promote_oldest(self, destination_region: torch.Tensor) -> int:
        """DDR -> HBM: promote the oldest staged bytes; returns the freed host slot."""
        if self._count == 0:
            raise RuntimeError("transit ring is empty; nothing to promote")
        host_slot = self._head
        with self._runtime.stream_context(self._dma_stream):
            destination_region.copy_(self._slot_regions[host_slot], non_blocking=True)
        self._states[host_slot] = RING_STATE_FREE
        self._head = (self._head + 1) % self._capacity
        self._count -= 1
        return host_slot

    def synchronize(self) -> None:
        self._runtime.synchronize_stream(self._dma_stream)
