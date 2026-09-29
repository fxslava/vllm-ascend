"""Static expert slot pool: AOT HBM arena with pre-sliced isomorphic views.

Memory contracts (enforced by the test suite):

1. Strict AOT residency -- every byte of device memory is allocated once in
   ``StaticExpertSlotPool.__init__``. The decode-step path performs no
   ``torch.empty``/``torch.zeros``/concat/clone at all.
2. Isomorphic expert slots -- all routed experts share one declarative
   ``ExpertTensorLayout`` (packed FP4 weights + E8M0 scales, block-32), and
   every slot is pre-sliced into per-parameter views at init so the hot loop
   never slices.
3. Transactional step locking -- the ``top_k`` slots requested for a step are
   locked before any eviction decision is taken; a step either commits fully
   or raises ``SlotExhaustionError`` with the pool state untouched.
4. "Lazy loading" is strictly ``slot_view.copy_(host_weight)`` into
   fixed-address views -- never module construction, never re-slicing, never
   reallocation. Blocking vs non-blocking transport is a *provider* property:
   DMA-capable host storage implements ``SlotFillProviderProtocol`` and fills
   the very same pre-sliced views itself on its own copy stream.

Device-notes for the NPU bring-up: the residency bookkeeping (LRU, locks) is
host-side control state; the ``expert_slot_table`` is a device-resident
``int32`` ``[num_layers, num_experts]`` tensor that attention/FFN kernels can
index directly. ``StaticExpertSlotPool.slot_of`` uses ``Tensor.item()`` and is
a *test/debug* accessor only -- hot paths must consume the table row on
device (AGENTS.md: no ``item()`` syncs in hot paths).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from ..core.config import DeepSeekV4MoEConfig
from ..core.layout import ExpertTensorLayout
from ..core.ledger import SlotExhaustionError, _AdmissionPlan, _SlotResidencyLedger
from ..protocols.provider import SlotFillProviderProtocol, WeightProviderProtocol

UNRESIDENT_SLOT_ID = -1

__all__ = [
    "SlotExhaustionError",
    "SlotPoolStats",
    "StepReservation",
    "StaticExpertSlotPool",
    "UNRESIDENT_SLOT_ID",
]


@dataclass
class SlotPoolStats:
    """Live diagnostics counters (mutated in place by the pool)."""

    hits: int = 0
    loads: int = 0
    evictions: int = 0
    bytes_staged: int = 0


@dataclass
class StepReservation:
    """Handle for one decode step's locked slots.

    Valid from ``acquire_for_step`` until ``release_step``; the pool is
    single-flight, i.e. ``device_slot_ids`` aliases a pool-owned AOT buffer
    that the next ``acquire_for_step`` overwrites.
    """

    layer_idx: int
    expert_ids: tuple[int, ...]
    slot_ids: tuple[int, ...]
    locked_slots: tuple[int, ...]
    device_slot_ids: torch.Tensor  # int32 [top_k], pool-owned, filled in place
    released: bool = False


class StaticExpertSlotPool:
    """AOT-allocated, zero-allocation residency manager for routed experts.

    Device-side consumers read ``expert_slot_table[layer_idx]`` (int32,
    -1 = miss) directly on device; ``weight_views(slot_id)`` /
    ``scale_views(slot_id)`` are the pre-sliced, fixed-address compute
    operands, and ``param_views(slot_id)`` is the same set keyed by provider
    param key for streamed DMA fills.
    """

    def __init__(
        self,
        config: DeepSeekV4MoEConfig,
        num_slots: int,
        layout: ExpertTensorLayout | None = None,
        device: str = "cpu",
    ):
        if num_slots < config.top_k:
            raise ValueError(f"num_slots {num_slots} must hold a full top-k of {config.top_k}")
        self._config = config
        self._layout = layout or ExpertTensorLayout.for_deepseek_v4_flash(config)
        self._num_slots = num_slots
        self._device = torch.device(device)

        # --- AOT physical allocation: the only allocating statements in this class. ---
        self._slot_arena = torch.empty(num_slots * self._layout.slot_num_bytes, dtype=torch.uint8, device=self._device)
        self._slot_arena.zero_()
        self._expert_slot_table = torch.full(
            (config.num_layers, config.num_routed_experts),
            UNRESIDENT_SLOT_ID,
            dtype=torch.int32,
            device=self._device,
        )
        self._step_slot_ids = torch.empty(config.top_k, dtype=torch.int32, device=self._device)

        # --- AOT view slicing: no slicing happens in the hot loop. ---
        self._weight_views: list[dict[str, torch.Tensor]] = []
        self._scale_views: list[dict[str, torch.Tensor]] = []
        self._param_views: list[dict[str, torch.Tensor]] = []
        for slot_id in range(num_slots):
            weights, scales = self._layout.slice_slot_views(self._slot_arena, slot_id)
            self._weight_views.append(weights)
            self._scale_views.append(scales)
            views_by_key: dict[str, torch.Tensor] = {}
            for spec in self._layout.specs:
                source_dict = weights if spec.kind == "packed_fp4" else scales
                views_by_key[spec.param_key] = source_dict[spec.name]
            self._param_views.append(views_by_key)

        # --- Host-side control state (no device memory). ---
        self._ledger = _SlotResidencyLedger(num_slots, config.num_routed_experts)
        self.stats = SlotPoolStats()

    # ------------------------------------------------------------------ views

    @property
    def slot_arena(self) -> torch.Tensor:
        """The monolithic slot storage; its data_ptr must never change."""
        return self._slot_arena

    @property
    def expert_slot_table(self) -> torch.Tensor:
        return self._expert_slot_table

    @property
    def step_slot_ids_buffer(self) -> torch.Tensor:
        return self._step_slot_ids

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def resident_expert_count(self) -> int:
        return self._ledger.resident_count()

    @property
    def layout(self) -> ExpertTensorLayout:
        return self._layout

    def weight_views(self, slot_id: int) -> Mapping[str, torch.Tensor]:
        return self._weight_views[slot_id]

    def scale_views(self, slot_id: int) -> Mapping[str, torch.Tensor]:
        return self._scale_views[slot_id]

    def param_views(self, slot_id: int) -> Mapping[str, torch.Tensor]:
        """Pre-sliced views keyed by provider param key ("w1", "w1_scale", ...)."""
        return self._param_views[slot_id]

    def slot_of(self, layer_idx: int, expert_id: int) -> int:
        """Test/debug accessor (``Tensor.item()``); hot paths read the table on device."""
        return int(self._expert_slot_table[layer_idx, expert_id].item())

    # ------------------------------------------------------- step acquisition

    def acquire_for_step(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        host_pinned_storage: WeightProviderProtocol,
    ) -> StepReservation:
        """Lock the step's top-k slots and make them resident, in place.

        Transactional: the full plan (hits, free reuse, evictions) is computed
        before any state mutates, so ``SlotExhaustionError`` leaves the pool
        untouched. Fills are strictly in-place ``copy_`` into pre-sliced views.
        """
        self._validate_request(layer_idx, expert_ids)
        keys = [
            _SlotResidencyLedger.expert_key(layer_idx, expert_id, self._config.num_routed_experts)
            for expert_id in expert_ids
        ]
        plan = self._ledger.plan_admission(keys)

        locked_slots = tuple(dict.fromkeys([slot for _, slot in plan.hits] + [slot for _, slot in plan.admissions]))
        self._ledger.lock_slots(locked_slots)
        try:
            self._apply_plan(layer_idx, plan, host_pinned_storage)
        except Exception:
            self._ledger.unlock_slots(locked_slots)
            raise

        slot_ids = tuple(plan.slot_for(key) for key in plan.request_order)
        for index, slot in enumerate(slot_ids):
            self._step_slot_ids[index] = slot
        return StepReservation(
            layer_idx=layer_idx,
            expert_ids=tuple(expert_ids),
            slot_ids=slot_ids,
            locked_slots=locked_slots,
            device_slot_ids=self._step_slot_ids,
        )

    @contextmanager
    def step(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        host_pinned_storage: WeightProviderProtocol,
    ) -> Iterator[StepReservation]:
        reservation = self.acquire_for_step(layer_idx, expert_ids, host_pinned_storage)
        try:
            yield reservation
        finally:
            self.release_step(reservation)

    def release_step(self, reservation: StepReservation) -> None:
        if reservation.released:
            raise ValueError("step reservation released twice")
        reservation.released = True
        self._ledger.unlock_slots(reservation.locked_slots)

    # -------------------------------------------------------------- internals

    def _validate_request(self, layer_idx: int, expert_ids: Sequence[int]) -> None:
        if not 0 <= layer_idx < self._config.num_layers:
            raise ValueError(f"layer_idx {layer_idx} outside [0, {self._config.num_layers})")
        if len(expert_ids) != self._config.top_k:
            raise ValueError(f"expected exactly top_k={self._config.top_k} experts, got {len(expert_ids)}")
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError(f"expert ids must be unique, got {expert_ids!r}")
        for expert_id in expert_ids:
            if not 0 <= expert_id < self._config.num_routed_experts:
                raise ValueError(f"expert_id {expert_id} outside [0, {self._config.num_routed_experts})")

    def _apply_plan(
        self,
        layer_idx: int,
        plan: _AdmissionPlan,
        host_pinned_storage: WeightProviderProtocol,
    ) -> None:
        # 1. Reserve and validate every source *before* mutating any state, so a
        #    misbehaving provider cannot leave the pool half-committed.
        uses_streamed_fill = isinstance(host_pinned_storage, SlotFillProviderProtocol)
        pending_fills: list[tuple[int, int, list[torch.Tensor]]] = []  # (expert_id, slot, sources)
        if uses_streamed_fill:
            for key, _slot in plan.admissions:
                _admitted_layer, expert_id = _SlotResidencyLedger.split_key(key, self._config.num_routed_experts)
                host_pinned_storage.ensure_staged(layer_idx, expert_id)
        else:
            for key, slot in plan.admissions:
                _admitted_layer, expert_id = _SlotResidencyLedger.split_key(key, self._config.num_routed_experts)
                sources = []
                for spec in self._layout.specs:
                    source = host_pinned_storage.pinned_cpu_weight(layer_idx, expert_id, spec.param_key)
                    self._assert_transfer_compatible(spec.param_key, source)
                    sources.append(source)
                pending_fills.append((expert_id, slot, sources))

        # 2. Bookkeeping (ledger + device table), then bytes. Copies cannot fail
        #    here: staging/shapes/dtypes were validated above.
        self._ledger.commit(plan)
        for victim_key, _slot in plan.evictions:
            victim_layer, victim_expert = _SlotResidencyLedger.split_key(victim_key, self._config.num_routed_experts)
            self._expert_slot_table[victim_layer, victim_expert] = UNRESIDENT_SLOT_ID
            self.stats.evictions += 1
        for key, slot in plan.hits:
            _hit_layer, expert_id = _SlotResidencyLedger.split_key(key, self._config.num_routed_experts)
            self._expert_slot_table[layer_idx, expert_id] = slot
            self.stats.hits += 1
        if uses_streamed_fill:
            for key, slot in plan.admissions:
                _admitted_layer, expert_id = _SlotResidencyLedger.split_key(key, self._config.num_routed_experts)
                self._expert_slot_table[layer_idx, expert_id] = slot
                self.stats.bytes_staged += host_pinned_storage.fill_slot_params(
                    layer_idx, expert_id, self._param_views[slot]
                )
                self.stats.loads += 1
        else:
            for expert_id, slot, sources in pending_fills:
                self._expert_slot_table[layer_idx, expert_id] = slot
                for spec, source in zip(self._layout.specs, sources):
                    view_map = self._weight_views if spec.kind == "packed_fp4" else self._scale_views
                    destination = view_map[slot][spec.name]
                    destination.copy_(source, non_blocking=False)
                    self.stats.bytes_staged += source.numel()
                self.stats.loads += 1

    def _assert_transfer_compatible(self, param_key: str, source: torch.Tensor) -> None:
        spec = self._layout.spec_for(param_key)
        if tuple(source.shape) != spec.view_shape:
            raise ValueError(
                f"{param_key}: provider returned shape {tuple(source.shape)}, slot view needs {spec.view_shape}"
            )
        if source.dtype != torch.uint8:
            raise ValueError(f"{param_key}: provider must hand over raw bytes (uint8), got {source.dtype}")
        if source.device.type != "cpu":
            raise ValueError(f"{param_key}: provider must serve host memory, got device {source.device}")
        if not source.is_contiguous():
            raise ValueError(f"{param_key}: provider tensor must be contiguous")
