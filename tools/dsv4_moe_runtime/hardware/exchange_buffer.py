"""Transit / exchange ring between HBM slots and pinned host DDR.

When the HBM slot pool must displace a resident expert, the victim's bytes are
staged into this bounded ring of pinned host slots *before* the incoming
expert overwrites the HBM slot (bi-directional exchange: victim HBM -> DDR on
one side, promotee DDR -> HBM on the other). The ring reuses its host slots
round-robin, so no host memory is ever (re)allocated at runtime: the arena,
its per-slot views and the ring state are built once at construction.

Zero-allocation contract: staging/consumption perform only in-place ``copy_``
into pre-sliced regions plus integer ring arithmetic; the ring never grows.

Two overflow disciplines (``overflow=``):

* ``"error"`` -- a full ring raises :class:`ExchangeBufferFullError`
  (back-pressure, never growth; the original contract and the default);
* ``"drop_oldest"`` -- exclusive non-inclusive staging: the oldest entry is
  reclaimed and its slot reused. This is safe exactly because the backing
  byte source is authoritative -- ring entries are droppable caches, never
  the only copy -- so the window stays bounded forever.

Keyed entries: staging with ``key=(layer_idx, expert_id)`` records identity so
``lookup`` finds recently evicted bytes (window hits) and ``consume_key``
hands them to a promotion. A consumed slot enters the ``INFLIGHT`` state (its
DMA into HBM may still be running) and is only reclaimed by
:meth:`synchronize`, which makes the region safe to overwrite.
"""

from __future__ import annotations

import torch

from ..core.layout import ExpertTensorLayout
from .runtime import DeviceRuntime

RING_STATE_FREE = 0
RING_STATE_DIRTY = 1  # holds staged bytes not yet promoted/consumed
RING_STATE_INFLIGHT = 2  # consumed; ring -> HBM DMA in flight until synchronize()

OVERFLOW_ERROR = "error"
OVERFLOW_DROP_OLDEST = "drop_oldest"
_VALID_OVERFLOW = (OVERFLOW_ERROR, OVERFLOW_DROP_OLDEST)


class ExchangeBufferFullError(RuntimeError):
    """Raised when every transit slot is occupied and none can be reclaimed."""


class TransitExchangeBuffer:
    """Bounded FIFO ring of pinned host slots for HBM <-> DDR expert exchange.

    Evictions push at the head (``stage_eviction``); keyed lookups and
    promotions drain from the tail, keeping evicted bytes resident in DDR for
    at least a full ring's worth of subsequent traffic.
    """

    def __init__(
        self,
        runtime: DeviceRuntime,
        layout: ExpertTensorLayout,
        host_slots: int,
        pin: bool | None = None,
        overflow: str = OVERFLOW_ERROR,
    ):
        if host_slots < 1:
            raise ValueError(f"host_slots must be >= 1, got {host_slots}")
        if overflow not in _VALID_OVERFLOW:
            raise ValueError(f"overflow must be one of {_VALID_OVERFLOW}, got {overflow!r}")
        self._runtime = runtime
        self._layout = layout
        self._capacity = host_slots
        self._overflow = overflow
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
        self._head = 0  # oldest occupied slot
        self._count = 0  # occupied transit slots (DIRTY + INFLIGHT)
        self._key_of_slot: list[object | None] = [None] * host_slots
        self._slot_of_key: dict[object, int] = {}
        # telemetry
        self.staged_evictions = 0
        self.consumed_entries = 0
        self.dropped_entries = 0

    # ---------------------------------------------------------------- views

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def capacity_bytes(self) -> int:
        """Total pinned host DDR held by the window (the staging bound)."""
        return self._layout.slot_num_bytes * self._capacity

    @property
    def overflow(self) -> str:
        return self._overflow

    @property
    def in_flight(self) -> int:
        return self._count

    @property
    def dma_stream(self) -> object | None:
        return self._dma_stream

    def state_of(self, host_slot: int) -> int:
        return int(self._states[host_slot])

    def key_of(self, host_slot: int) -> object | None:
        return self._key_of_slot[host_slot]

    def host_region(self, host_slot: int) -> torch.Tensor:
        """Pre-sliced pinned region of one transit slot (fixed address)."""
        return self._slot_regions[host_slot]

    def lookup(self, key: object) -> int | None:
        """Host slot currently holding the key's staged bytes, or None."""
        return self._slot_of_key.get(key)

    def fingerprint(self) -> list[int]:
        """data_ptr of the arena and every slot region (host AOT invariant)."""
        return [self._arena.data_ptr()] + [region.data_ptr() for region in self._slot_regions]

    # ------------------------------------------------------------- transfers

    def stage_eviction(self, source_region: torch.Tensor, key: object = None) -> int:
        """HBM -> DDR: stage one evicted expert's slot region; returns host slot."""
        self._assert_slot_region(source_region)
        host_slot = self._allocate(key)
        with self._runtime.stream_context(self._dma_stream):
            self._slot_regions[host_slot].copy_(source_region, non_blocking=True)
        self.staged_evictions += 1
        return host_slot

    def acquire_fill_slot(self, key: object) -> torch.Tensor:
        """Reserve a window slot for an inbound streamed fill; returns the region.

        The caller writes the authoritative bytes into the returned region
        before any other buffer call, which makes the entry visible to
        ``lookup``. Superseding an already-staged key is a caller bug.
        """
        if key in self._slot_of_key:
            raise ValueError(f"key {key!r} is already staged in the transit window")
        return self._slot_regions[self._allocate(key)]

    def consume_key(self, key: object) -> int:
        """Mark a staged entry consumed (promotion in flight); returns its slot.

        The caller enqueues the ring -> HBM copies *before* consuming. The slot
        stays occupied (INFLIGHT) until :meth:`synchronize` reclaims it, so the
        DMA source cannot be overwritten early.
        """
        host_slot = self._slot_of_key.pop(key, None)
        if host_slot is None:
            raise KeyError(f"key {key!r} is not staged in the transit window")
        self._states[host_slot] = RING_STATE_INFLIGHT
        self._key_of_slot[host_slot] = None
        self.consumed_entries += 1
        return host_slot

    def promote_oldest(self, destination_region: torch.Tensor) -> int:
        """DDR -> HBM: copy the oldest staged bytes out; frees the host slot."""
        if self._count == 0:
            raise RuntimeError("transit ring is empty; nothing to promote")
        if self.state_of(self._head) == RING_STATE_INFLIGHT:
            raise RuntimeError("oldest transit entry is in flight; synchronize first")
        host_slot = self._head
        with self._runtime.stream_context(self._dma_stream):
            destination_region.copy_(self._slot_regions[host_slot], non_blocking=True)
        self._release_oldest()
        return host_slot

    def synchronize(self) -> None:
        """Sync the DMA stream, then reclaim head-adjacent consumed slots."""
        self._runtime.synchronize_stream(self._dma_stream)
        while self._count and self.state_of(self._head) == RING_STATE_INFLIGHT:
            self._release_oldest()

    # ------------------------------------------------------------- internals

    def _assert_slot_region(self, source_region: torch.Tensor) -> None:
        if source_region.numel() != self._layout.slot_num_bytes:
            raise ValueError(
                f"staged region holds {source_region.numel()} bytes, transit expects {self._layout.slot_num_bytes}"
            )

    def _allocate(self, key: object) -> int:
        if self._count == self._capacity:
            if self._overflow != OVERFLOW_DROP_OLDEST:
                raise ExchangeBufferFullError(
                    f"transit ring exhausted ({self._capacity} slots in flight); "
                    "promote, consume or synchronize before staging further entries"
                )
            self._make_headroom()
        host_slot = (self._head + self._count) % self._capacity
        self._count += 1
        self._states[host_slot] = RING_STATE_DIRTY
        self._forget(key)
        self._key_of_slot[host_slot] = key
        if key is not None:
            self._slot_of_key[key] = host_slot
        return host_slot

    def _make_headroom(self) -> None:
        """Drop-oldest mode: reclaim exactly one slot for the pending allocate.

        An INFLIGHT head cannot be overwritten (its DMA into HBM may still be
        running), so it is sync-released first; the walk may free several
        entries, after which at most one DIRTY head is dropped.
        """
        if self.state_of(self._head) == RING_STATE_INFLIGHT:
            self.synchronize()
        if self._count == self._capacity:
            self._release_oldest()
            self.dropped_entries += 1

    def _release_oldest(self) -> None:
        key = self._key_of_slot[self._head]
        if key is not None:
            self._slot_of_key.pop(key, None)
        self._key_of_slot[self._head] = None
        self._states[self._head] = RING_STATE_FREE
        self._head = (self._head + 1) % self._capacity
        self._count -= 1

    def _forget(self, key: object) -> None:
        """Drop a key's index entry (its old slot ages out anonymously)."""
        old_slot = self._slot_of_key.pop(key, None)
        if old_slot is not None:
            self._key_of_slot[old_slot] = None
