"""Exclusive staging provider: the bounded transit window as the fill source.

``ExclusiveStagingProvider`` implements ``SlotFillProviderProtocol`` against a
:class:`~tools.dsv4_moe_runtime.hardware.exchange_buffer.TransitExchangeBuffer`
instead of a monolithic host arena. No expert ever owns a permanent host copy:
"staged" means *the bytes currently sit somewhere inside the bounded window*,
and a window miss streams the expert in from the authoritative
:class:`~tools.dsv4_moe_runtime.protocols.provider.WeightByteSource` on the
fly. Total pinned host memory is therefore the window capacity -- one
configurable transit bound -- regardless of how many layers or steps run.

Contracts (``SlotFillProviderProtocol``):

* ``ensure_staged`` raises ``KeyError`` *before* any byte moves or window
  state mutates when the expert is outside the source's coverage;
* ``fill_slot_params`` promotes the window entry into the pool's pre-sliced
  views on the window's DMA stream and then *consumes* the entry -- exclusive,
  non-inclusive: after ``synchronize`` the bytes live in HBM only. If a
  ``drop_oldest`` ring reclaimed the entry between staging and the fill, the
  bytes are re-read from the source (see :meth:`_restage`);
* host arena fingerprints never move (the window is allocated AOT).

Window entries staged by an acquisition that later fails are droppable garbage
(the pool commits nothing), never authoritative -- the source always is.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ..core.layout import ExpertTensorLayout
from ..protocols.provider import WeightByteSource
from .exchange_buffer import TransitExchangeBuffer
from .runtime import DeviceRuntime


class ExclusiveStagingProvider:
    """Streams experts through a bounded transit window (SlotFillProvider)."""

    def __init__(
        self,
        runtime: DeviceRuntime,
        layout: ExpertTensorLayout,
        exchange_buffer: TransitExchangeBuffer,
        source: WeightByteSource,
    ):
        self._runtime = runtime
        self._layout = layout
        self._buffer = exchange_buffer
        self._source = source
        # Keys ``ensure_staged`` prepared for the in-flight step: only these may
        # be re-materialized by :meth:`_restage`; anything else reaching
        # ``fill_slot_params`` unstaged is a caller contract violation.
        self._prepared: set[tuple[int, int]] = set()
        # Shares the window's DMA stream: per-step program order makes eviction
        # staging (HBM -> window) complete before a promotion into the same
        # HBM slot, and vice versa, without extra events.
        self._dma_stream = exchange_buffer.dma_stream
        self.bytes_copied = 0
        self.dma_copy_count = 0
        self.window_source_fills = 0
        self.window_hits = 0  # entries found in the window at ensure_staged time
        self.window_refills = 0  # of those, the ones the ring reclaimed before the fill

    # ----------------------------------------------------------- measurement

    @property
    def window_slots(self) -> int:
        return self._buffer.capacity

    @property
    def window_bytes(self) -> int:
        """Pinned host DDR held by the window (the exclusive staging bound)."""
        return self._buffer.capacity_bytes

    @property
    def window_in_flight(self) -> int:
        return self._buffer.in_flight

    # ------------------------------------------------------------ SlotFill

    def ensure_staged(self, layer_idx: int, expert_id: int) -> None:
        if not self._source.contains(layer_idx, expert_id):
            raise KeyError(f"expert (layer={layer_idx}, id={expert_id}) is outside the byte source coverage")
        key = (layer_idx, expert_id)
        self._prepared.add(key)  # this step may re-materialize it if the ring reclaims it
        if self._buffer.lookup(key) is not None:
            self.window_hits += 1  # recently evicted bytes are still in the window
            return
        region = self._buffer.acquire_fill_slot(key)
        self._source.fill_slot(region, layer_idx, expert_id)
        self.window_source_fills += 1

    def fill_slot_params(self, layer_idx: int, expert_id: int, views: Mapping[str, torch.Tensor]) -> int:
        key = (layer_idx, expert_id)
        host_slot = self._buffer.lookup(key)
        if host_slot is None:
            host_slot = self._restage(layer_idx, expert_id)
        region = self._buffer.host_region(host_slot)
        moved = 0
        with self._runtime.stream_context(self._dma_stream):
            for param_key, destination in views.items():
                destination.copy_(self._param_region(region, param_key), non_blocking=True)
                moved += destination.numel()
        self._buffer.consume_key(key)  # exclusive: window copy dies with the promotion
        self._prepared.discard(key)
        self.bytes_copied += moved
        self.dma_copy_count += len(views)
        return moved

    def pinned_cpu_weight(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor:
        host_slot = self._buffer.lookup((layer_idx, expert_id))
        if host_slot is None:
            raise KeyError(f"expert (layer={layer_idx}, id={expert_id}) is not staged in the transit window")
        return self._param_region(self._buffer.host_region(host_slot), param_key)

    def synchronize(self) -> None:
        self._buffer.synchronize()
        self._prepared.clear()  # the step is over: nothing is promotable any more

    # ------------------------------------------------------------- internals

    def _restage(self, layer_idx: int, expert_id: int) -> int:
        """Re-materialize an entry the bounded ring reclaimed before its fill.

        ``ensure_staged`` guarantees the bytes are *in* the window, not that a
        ``drop_oldest`` ring keeps them there: the same step's eviction staging
        allocates further window slots, and in a full ring those allocations
        reclaim the oldest entries -- which is exactly where a window *hit* on a
        long-evicted expert sits. Re-reading the authoritative source is always
        correct (window entries are droppable caches, never the only copy) and
        keeps the step transactional instead of failing mid-admission.
        """
        key = (layer_idx, expert_id)
        if key not in self._prepared:
            raise KeyError(f"expert (layer={layer_idx}, id={expert_id}) is not staged in the transit window")
        region = self._buffer.acquire_fill_slot(key)
        self._source.fill_slot(region, layer_idx, expert_id)
        self.window_refills += 1
        host_slot = self._buffer.lookup(key)
        if host_slot is None:  # pragma: no cover - the ring just staged this key
            raise RuntimeError(f"transit window lost expert (layer={layer_idx}, id={expert_id}) during restaging")
        return host_slot

    def _param_region(self, region: torch.Tensor, param_key: str) -> torch.Tensor:
        spec = self._layout.spec_for(param_key)
        return region[spec.offset_bytes : spec.offset_bytes + spec.num_bytes].view(spec.view_shape)
