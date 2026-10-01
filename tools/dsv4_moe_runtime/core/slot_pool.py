"""Static expert slot pool: AOT HBM arena with pre-sliced isomorphic views.

Responsibility split (SRP): this class owns *physical* memory mechanics only
-- arena allocation, view slicing, slot indexing, step locking and DMA fills.
*Which* resident expert to evict and how priorities are computed belong to the
injected
:class:`~tools.dsv4_moe_runtime.protocols.residency_policy.EvictionPolicyProtocol`
(default:
:class:`~tools.dsv4_moe_runtime.core.generational_policy.GenerationalRadixPolicy`;
the historical flat-LRU behaviour remains available as
:class:`~tools.dsv4_moe_runtime.core.legacy_lru_policy.LegacyLruPolicy`).

Memory contracts (enforced by the test suite):

1. Strict AOT residency -- every byte of device memory is allocated once in
   ``StaticExpertSlotPool.__init__``. The decode-step path performs no
   ``torch.empty``/``torch.zeros``/concat/clone at all.
2. Isomorphic expert slots -- all routed experts share one declarative
   ``ExpertTensorLayout`` (packed FP4 weights + E8M0 scales, block-32), and
   every slot is pre-sliced into per-parameter views at init so the hot loop
   never slices.
3. Transactional step locking -- the ``top_k`` slots requested for a step are
   locked before any state mutates; a step either commits fully or raises
   ``SlotExhaustionError`` with the pool and policy untouched.
4. "Lazy loading" is strictly ``slot_view.copy_(host_weight)`` into
   fixed-address views. Blocking vs non-blocking transport is a *provider*
   property: DMA-capable host storage implements ``SlotFillProviderProtocol``
   and fills the very same pre-sliced views itself on its own copy stream.

Device-notes for the NPU bring-up: ``expert_slot_table`` is a device-resident
``int32`` ``[num_layers, num_experts]`` tensor that kernels index directly;
``slot_of`` uses ``Tensor.item()`` and is a *test/debug* accessor only.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from ..core.config import DeepSeekV4MoEConfig
from ..core.generational_policy import GenerationalRadixPolicy
from ..core.layout import ExpertTensorLayout
from ..hardware.exchange_buffer import TransitExchangeBuffer
from ..protocols.provider import SlotFillProviderProtocol, WeightProviderProtocol
from ..protocols.residency_policy import EvictionPolicyProtocol

UNRESIDENT_SLOT_ID = -1

__all__ = [
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

    Eviction strategy is injected via ``policy``; the pool applies the returned
    :class:`~tools.dsv4_moe_runtime.protocols.residency_policy.AdmissionDecision`
    to physical state (locks, residency table, DMA fills, free list). An
    optional ``exchange_buffer`` stages evicted slot bytes into the pinned-DDR
    transit ring before the incoming expert overwrites the slot.
    """

    def __init__(
        self,
        config: DeepSeekV4MoEConfig,
        num_slots: int,
        layout: ExpertTensorLayout | None = None,
        device: str = "cpu",
        policy: EvictionPolicyProtocol | None = None,
        exchange_buffer: TransitExchangeBuffer | None = None,
        shared_layout: ExpertTensorLayout | None = None,
        shared_layers: Sequence[int] = (),
    ):
        if num_slots < config.top_k:
            raise ValueError(f"num_slots {num_slots} must hold a full top-k of {config.top_k}")
        self._config = config
        self._layout = layout or ExpertTensorLayout.for_deepseek_v4_flash(config)
        self._num_slots = num_slots
        self._device = torch.device(device)
        self._policy = policy or GenerationalRadixPolicy(config, num_slots, device=device)
        self._exchange_buffer = exchange_buffer

        # --- Shared-expert reservation, sized before the arena is allocated. ---
        # A shared expert is not routed: every token at every MoE layer uses it,
        # so it is reserved rather than admitted. Letting the eviction policy
        # see it would be strictly wasted work -- it can never be the right
        # victim, and it would be re-admitted on the very next layer. The
        # reservation sits *above* the routed slots, so routed ids stay
        # 0..num_slots-1 and the policy's view of the pool is unchanged.
        self._shared_layout = shared_layout
        self._shared_layers = tuple(dict.fromkeys(shared_layers))
        if self._shared_layout is None:
            if self._shared_layers:
                raise ValueError("shared_layers given without a shared_layout to place them with")
            self._slots_per_shared = 0
        else:
            self._slots_per_shared = self._shared_layout.slots_per_region(self._layout)
            for layer_idx in self._shared_layers:
                if not 0 <= layer_idx < config.num_layers:
                    raise ValueError(f"shared layer {layer_idx} outside [0, {config.num_layers})")
        self._reserved_slots = len(self._shared_layers) * self._slots_per_shared
        total_slots = num_slots + self._reserved_slots

        # --- AOT physical allocation: the only allocating statements in this class. ---
        self._slot_arena = torch.empty(
            total_slots * self._layout.slot_num_bytes, dtype=torch.uint8, device=self._device
        )
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
                source_dict = scales if spec.is_scale else weights
                views_by_key[spec.param_key] = source_dict[spec.name]
            self._param_views.append(views_by_key)

        # --- Shared-expert views: one paired region per layer, sliced AOT. ---
        self._shared_slot_ids: dict[int, tuple[int, ...]] = {}
        self._shared_views: dict[int, dict[str, torch.Tensor]] = {}
        self._shared_filled: set[int] = set()
        for index, layer_idx in enumerate(self._shared_layers):
            first = num_slots + index * self._slots_per_shared
            self._shared_slot_ids[layer_idx] = tuple(range(first, first + self._slots_per_shared))
            assert self._shared_layout is not None  # guarded above
            self._shared_views[layer_idx] = self._shared_layout.slice_region_views(
                self._slot_arena, first * self._layout.slot_num_bytes
            )

        # --- Physical residency bookkeeping (no device memory). ---
        # Lock counts cover the reserved slots too, so a stray unlock of one is
        # an error rather than an index out of range; the free list does not,
        # which is what keeps a reserved slot from ever being admitted into.
        self._residents: dict[tuple[int, int], int] = {}
        self._lock_counts: list[int] = [0] * total_slots
        self._free_slots: list[int] = list(range(num_slots))
        self._current_token = 0
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
    def policy(self) -> EvictionPolicyProtocol:
        """The injected eviction strategy (swappable at construction, OCP/DIP)."""
        return self._policy

    @property
    def resident_expert_count(self) -> int:
        return len(self._residents)

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

    def slot_region(self, slot_id: int) -> torch.Tensor:
        """The whole raw ``slot_num_bytes`` region of one slot (transit staging)."""
        return self._slot_arena.narrow(0, slot_id * self._layout.slot_num_bytes, self._layout.slot_num_bytes)

    # ------------------------------------------------------- shared experts

    @property
    def shared_layout(self) -> ExpertTensorLayout | None:
        return self._shared_layout

    @property
    def shared_layers(self) -> tuple[int, ...]:
        """MoE layers whose shared expert has a reserved slot pair."""
        return self._shared_layers

    @property
    def slots_per_shared_expert(self) -> int:
        """Routed slots one shared expert occupies (2 on DeepSeek-V2-Lite)."""
        return self._slots_per_shared

    @property
    def reserved_slot_count(self) -> int:
        return self._reserved_slots

    @property
    def total_slot_count(self) -> int:
        """Routed plus reserved; what the arena is actually sized for."""
        return self._num_slots + self._reserved_slots

    def shared_slot_ids(self, layer_idx: int) -> tuple[int, ...]:
        """The consecutive slot ids backing one layer's shared expert."""
        try:
            return self._shared_slot_ids[layer_idx]
        except KeyError:
            raise KeyError(
                f"layer {layer_idx} has no shared-expert reservation; reserved layers: {list(self._shared_slot_ids)}"
            ) from None

    def shared_param_views(self, layer_idx: int) -> Mapping[str, torch.Tensor]:
        """Pre-sliced shared-expert views, keyed as the provider fills them."""
        self.shared_slot_ids(layer_idx)  # raises with the helpful message
        return self._shared_views[layer_idx]

    def shared_region(self, layer_idx: int) -> torch.Tensor:
        """The whole raw byte region of one layer's shared expert."""
        first = self.shared_slot_ids(layer_idx)[0]
        assert self._shared_layout is not None
        return self._slot_arena.narrow(0, first * self._layout.slot_num_bytes, self._shared_layout.slot_num_bytes)

    def shared_expert_is_resident(self, layer_idx: int) -> bool:
        return layer_idx in self._shared_filled

    def fill_shared_experts(self, provider: object) -> int:
        """Stream every reserved layer's shared expert in, once, and report bytes.

        Called at warm-up rather than per step: the reservation is permanent, so
        the fill is too. Re-filling is allowed (a provider swap) but tracked, so
        a dispatcher can refuse to compute against a region nothing has written
        -- zeroed BF16 is a silent wrong answer, not a crash.
        """
        if self._shared_layout is None:
            return 0
        filler = getattr(provider, "fill_shared_expert", None)
        if filler is None:
            raise TypeError(
                f"{type(provider).__name__} cannot serve shared experts: no fill_shared_expert method. "
                "Construct the pool without shared_layers, or use a provider that implements it."
            )
        staged = 0
        for layer_idx in self._shared_layers:
            staged += int(filler(layer_idx, self._shared_views[layer_idx]))
            self._shared_filled.add(layer_idx)
        self.stats.bytes_staged += staged
        return staged

    def slot_of(self, layer_idx: int, expert_id: int) -> int:
        """Test/debug accessor (``Tensor.item()``); hot paths read the table on device."""
        return int(self._expert_slot_table[layer_idx, expert_id].item())

    # ------------------------------------------------------- token boundaries

    def advance_generation(self, completed_token_idx: int) -> None:
        """Close out a token (idempotent); ages the policy's generational state."""
        self._policy.advance_generation(completed_token_idx)
        self._current_token = completed_token_idx + 1

    # ------------------------------------------------------- step acquisition

    def acquire_for_step(
        self,
        layer_idx: int,
        expert_ids: Sequence[int],
        host_pinned_storage: WeightProviderProtocol,
    ) -> StepReservation:
        """Lock the step's top-k slots and make them resident, in place.

        Transactional: the policy plans the full decision (hits, free reuse,
        evictions) before any state mutates, and provider sources are validated
        before the policy commits -- a failure leaves everything untouched.
        """
        self._validate_request(layer_idx, expert_ids)
        requested_pairs = [(layer_idx, expert_id) for expert_id in expert_ids]
        for pair in requested_pairs:
            self._policy.register_access(pair[0], pair[1], self._current_token)

        locked_slots = [slot for slot, count in enumerate(self._lock_counts) if count]
        decision = self._policy.plan_admissions(
            requested_pairs, self._residents, self._free_slots, locked_slots=locked_slots
        )
        self._validate_admission_sources(layer_idx, decision, host_pinned_storage)

        self._policy.commit_decision(decision)
        locked = tuple(dict.fromkeys([slot for _, slot in decision.hits] + [slot for _, slot in decision.admissions]))
        self._lock_slots(locked)
        try:
            self._apply_decision(layer_idx, decision, host_pinned_storage)
        except Exception:
            self._unlock_slots(locked)
            raise

        slot_ids = tuple(decision.slot_for(pair) for pair in requested_pairs)
        for index, slot in enumerate(slot_ids):
            self._step_slot_ids[index] = slot
        return StepReservation(
            layer_idx=layer_idx,
            expert_ids=tuple(expert_ids),
            slot_ids=slot_ids,
            locked_slots=locked,
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
        self._unlock_slots(reservation.locked_slots)

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

    def _validate_admission_sources(
        self,
        layer_idx: int,
        decision,
        host_pinned_storage: WeightProviderProtocol,
    ) -> None:
        """Fetch/validate (or reserve) every source before any state mutates."""
        if isinstance(host_pinned_storage, SlotFillProviderProtocol):
            for key, _slot in decision.admissions:
                host_pinned_storage.ensure_staged(layer_idx, key[1])
            return
        for key, _slot in decision.admissions:
            for spec in self._layout.specs:
                source = host_pinned_storage.pinned_cpu_weight(layer_idx, key[1], spec.param_key)
                self._assert_transfer_compatible(spec.param_key, source)

    def _apply_decision(
        self,
        layer_idx: int,
        decision,
        host_pinned_storage: WeightProviderProtocol,
    ) -> None:
        # 1. Transit staging: evicted slot bytes go to the pinned-DDR ring
        #    before the incoming expert overwrites the slot. The victim key is
        #    recorded so a re-request can hit the window instead of the source.
        if self._exchange_buffer is not None:
            for victim_key, victim_slot in decision.evictions:
                self._exchange_buffer.stage_eviction(self.slot_region(victim_slot), key=victim_key)

        # 2. Physical residency bookkeeping (residents, free list, device table).
        for key, victim_slot in decision.evictions:
            del self._residents[key]
            self._expert_slot_table[key[0], key[1]] = UNRESIDENT_SLOT_ID
            self.stats.evictions += 1
        for key, slot in decision.hits:
            self._residents[key] = slot
            self._expert_slot_table[key[0], key[1]] = slot
            self.stats.hits += 1
        if decision.free_slot_count:
            del self._free_slots[-decision.free_slot_count :]

        # 3. Bytes: schema-driven fills, no per-projection duplication.
        if isinstance(host_pinned_storage, SlotFillProviderProtocol):
            for key, slot in decision.admissions:
                self._expert_slot_table[layer_idx, key[1]] = slot
                self._residents[key] = slot
                self.stats.bytes_staged += host_pinned_storage.fill_slot_params(
                    layer_idx, key[1], self._param_views[slot]
                )
                self.stats.loads += 1
        else:
            for key, slot in decision.admissions:
                self._expert_slot_table[layer_idx, key[1]] = slot
                self._residents[key] = slot
                for spec in self._layout.specs:
                    source = host_pinned_storage.pinned_cpu_weight(layer_idx, key[1], spec.param_key)
                    view_map = self._scale_views if spec.is_scale else self._weight_views
                    destination = view_map[slot][spec.name]
                    destination.copy_(source, non_blocking=False)
                    self.stats.bytes_staged += source.numel()
                self.stats.loads += 1

    def _lock_slots(self, slots: Sequence[int]) -> None:
        for slot in slots:
            self._lock_counts[slot] += 1

    def _unlock_slots(self, slots: Sequence[int]) -> None:
        for slot in slots:
            if self._lock_counts[slot] <= 0:
                raise ValueError(f"slot {slot} unlocked too often")
            self._lock_counts[slot] -= 1

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
