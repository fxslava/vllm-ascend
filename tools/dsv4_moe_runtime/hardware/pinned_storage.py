"""Pinned host DDR staging with a dedicated DMA copy stream.

``AscendPinnedHostStorage`` is the production-shaped
``SlotFillProviderProtocol`` implementation: one monolithic page-locked host
allocation, sliced AOT by ``ExpertTensorLayout``, streamed into slot views
with in-place ``copy_(..., non_blocking=True)``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..core.layout import ExpertTensorLayout
from ..hardware.runtime import DeviceRuntime


class AscendPinnedHostStorage:
    """Stages every routed expert in page-locked host memory (SlotFillProvider).

    No host allocation happens at serve time: the arena and its per-expert
    parameter views are built once at init. DMA transfers are in-place
    ``slot_view.copy_(pinned, non_blocking=True)`` on a dedicated copy stream;
    the caller synchronizes via :meth:`synchronize`.
    """

    def __init__(
        self,
        runtime: DeviceRuntime,
        layout: ExpertTensorLayout,
        layer_ids: Sequence[int],
        experts_per_layer: int,
        pin: bool | None = None,
    ):
        self._runtime = runtime
        self._layout = layout
        self._layer_ids = tuple(layer_ids)
        self._experts_per_layer = experts_per_layer
        self._dma_stream = runtime.make_stream()
        self.bytes_copied = 0
        self.dma_copy_count = 0
        use_pin = runtime.supports_pinned_host_memory if pin is None else pin
        if use_pin and not runtime.supports_pinned_host_memory:
            raise ValueError("pinned host memory requested but unsupported by this runtime")
        self.pinned = use_pin
        arena_flags: dict[str, object] = {"pin_memory": True} if use_pin else {}
        self._arena = torch.empty(
            layout.slot_num_bytes * len(self._layer_ids) * experts_per_layer, dtype=torch.uint8, **arena_flags
        )
        self._params: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        for host_slot, layer_idx in enumerate(self._layer_ids):
            for expert_id in range(experts_per_layer):
                weights, scales = layout.slice_slot_views(self._arena, host_slot * experts_per_layer + expert_id)
                views: dict[str, torch.Tensor] = {}
                for spec in layout.specs:
                    source_dict = weights if spec.kind == "packed_fp4" else scales
                    views[spec.param_key] = source_dict[spec.name]
                self._params[(layer_idx, expert_id)] = views

    @property
    def dma_stream(self) -> object | None:
        return self._dma_stream

    @property
    def staged_expert_count(self) -> int:
        return len(self._params)

    def staged_bytes(self) -> int:
        return self._arena.numel()

    def ensure_staged(self, layer_idx: int, expert_id: int) -> None:
        if (layer_idx, expert_id) not in self._params:
            raise KeyError(f"expert (layer={layer_idx}, id={expert_id}) is not staged in host storage")

    def pinned_cpu_weight(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor:
        self.ensure_staged(layer_idx, expert_id)
        return self._params[(layer_idx, expert_id)][param_key]

    def fill_slot_params(self, layer_idx: int, expert_id: int, views: Mapping[str, torch.Tensor]) -> int:
        source_views = self._params[(layer_idx, expert_id)]
        moved = 0
        with self._runtime.stream_context(self._dma_stream):
            for param_key, destination in views.items():
                destination.copy_(source_views[param_key], non_blocking=True)
                moved += destination.numel()
        self.bytes_copied += moved
        self.dma_copy_count += len(views)
        return moved

    def synchronize(self) -> None:
        self._runtime.synchronize_stream(self._dma_stream)
