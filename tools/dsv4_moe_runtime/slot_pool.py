"""Static expert slot pool for the DeepSeek-V4 Flash MoE offload runtime (pilot).

DeepSeek-V4 Flash (43 blocks, 256 routed experts, top-6) carries ~150 GiB of
block-32 FP4 routed-expert weights, which exceeds the 128 GiB HBM of a single
Ascend 950PR. The runtime therefore keeps routed experts in pinned DDR and
serves them through a fixed set of *isomorphic* HBM slots that are allocated
once, ahead of time, and refreshed strictly in place.

Pilot scope (this module runs on CPU with synthetic tensors): it owns the
*memory architecture* only -- declarative layout, arena allocation, view
slicing, LRU residency and transactional step-level locking. The on-device
follow-up binds the very same byte movements to aclnn V5 entry points
(Ascend 950PR rejects aclnn V1-V4 with EZ9903); every transfer is therefore
expressed as a single in-place ``copy_`` into a pre-sliced view, which maps
1:1 onto an aclnn copy plan with a pre-allocated workspace.

Memory contracts (enforced by ``test_pilot_memory.py``):

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
4. "Lazy loading" is strictly ``slot_view.copy_(pinned_cpu_weight,
   non_blocking=False)`` into fixed-address views -- never module construction,
   never re-slicing, never reallocation.

Device-notes for the NPU bring-up: the residency bookkeeping (LRU, locks) is
host-side control state; the ``expert_slot_table`` is a device-resident
``int32`` ``[num_layers, num_experts]`` tensor that attention/FFN kernels can
index directly. ``StaticExpertSlotPool.slot_of`` uses ``Tensor.item()`` and is
a *test/debug* accessor only -- hot paths must consume the table row on
device (AGENTS.md: no ``item()`` syncs in hot paths).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

EXPERT_PARAM_NAMES: tuple[str, ...] = ("w1", "w2", "w3")
SCALE_PARAM_SUFFIX = "_scale"
FP4_ELEMS_PER_BYTE = 2
FP4_BLOCK_SIZE = 32
E8M0_SCALE_NUM_BYTES = 1
SLOT_REGION_ALIGN_BYTES = 128
UNRESIDENT_SLOT_ID = -1


class SlotExhaustionError(RuntimeError):
    """Raised when a step cannot lock all of its top-k slots without evicting a locked slot."""


@dataclass(frozen=True)
class DeepSeekV4MoEConfig:
    """Routed-expert geometry, mirroring ``C:\\DeepSeekV4\\config.json``.

    ``num_layers`` counts the decoder blocks that own routed experts (the MTP
    block, ``mtp.0``, owns another identical MoE and is served by giving this
    pool ``num_layers=44`` or by a second pool instance).
    """

    hidden_size: int = 4096
    moe_intermediate_size: int = 2048
    num_routed_experts: int = 256
    top_k: int = 6
    num_layers: int = 43
    num_hash_layers: int = 3
    vocab_size: int = 129280


@dataclass(frozen=True)
class ExpertTensorSpec:
    """Byte-level geometry of one expert parameter region inside a slot.

    ``cols`` is always the *logical* FP4 element count along the reduction dim;
    storage compaction (2 nibbles/byte) or block scaling (1 E8M0 byte per 32
    elements) is derived, never hand-written per tensor.
    """

    name: str
    kind: str  # "packed_fp4" or "e8m0_scale"
    rows: int
    cols: int
    offset_bytes: int

    @property
    def stored_cols(self) -> int:
        if self.kind == "packed_fp4":
            if self.cols % FP4_ELEMS_PER_BYTE != 0:
                raise ValueError(f"{self.name}: logical cols {self.cols} not divisible by {FP4_ELEMS_PER_BYTE}")
            return self.cols // FP4_ELEMS_PER_BYTE
        if self.kind == "e8m0_scale":
            if self.cols % FP4_BLOCK_SIZE != 0:
                raise ValueError(f"{self.name}: logical cols {self.cols} not divisible by block {FP4_BLOCK_SIZE}")
            return self.cols // FP4_BLOCK_SIZE
        raise ValueError(f"unknown tensor kind: {self.kind}")

    @property
    def num_bytes(self) -> int:
        return self.rows * self.stored_cols * E8M0_SCALE_NUM_BYTES

    @property
    def view_shape(self) -> tuple[int, int]:
        return (self.rows, self.stored_cols)

    @property
    def param_key(self) -> str:
        """Key used with :class:`WeightProviderProtocol`."""
        return self.name if self.kind == "packed_fp4" else self.name + SCALE_PARAM_SUFFIX


@dataclass(frozen=True)
class ExpertTensorLayout:
    """Declarative schema for one isomorphic expert slot.

    Drives arena sizing, per-slot view slicing and the schema-driven DMA loop,
    so w1/w2/w3 never get hand-copied anywhere (DRY).
    """

    specs: tuple[ExpertTensorSpec, ...]
    slot_num_bytes: int

    @classmethod
    def for_deepseek_v4_flash(cls, config: DeepSeekV4MoEConfig) -> ExpertTensorLayout:
        inter = config.moe_intermediate_size
        hidden = config.hidden_size
        projection_shapes: dict[str, tuple[int, int]] = {
            "w1": (inter, hidden),
            "w2": (hidden, inter),
            "w3": (inter, hidden),
        }
        specs: list[ExpertTensorSpec] = []
        cursor = 0
        for name in EXPERT_PARAM_NAMES:
            rows, cols = projection_shapes[name]
            for kind in ("packed_fp4", "e8m0_scale"):
                spec = ExpertTensorSpec(name=name, kind=kind, rows=rows, cols=cols, offset_bytes=_align_up(cursor))
                specs.append(spec)
                cursor = spec.offset_bytes + spec.num_bytes
        return cls(specs=tuple(specs), slot_num_bytes=_align_up(cursor))

    def spec_for(self, param_key: str) -> ExpertTensorSpec:
        for spec in self.specs:
            if spec.param_key == param_key:
                return spec
        raise KeyError(f"layout has no parameter {param_key!r}; known: {[s.param_key for s in self.specs]}")

    def slice_slot_views(
        self, arena: torch.Tensor, slot_id: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Pre-slice one slot into (packed weights, e8m0 scales) views.

        Called once per slot at init; the decode loop only *reads* the dicts.
        Views are ``uint8`` byte windows (itemsize-1 storage of both
        ``float4_e2m1fn_x2`` and ``float8_e8m0fnu``); compute layers reinterpret
        them with ``Tensor.view(dtype)`` at bind time.
        """
        slot_base = arena.narrow(0, slot_id * self.slot_num_bytes, self.slot_num_bytes)
        weights: dict[str, torch.Tensor] = {}
        scales: dict[str, torch.Tensor] = {}
        for spec in self.specs:
            region = slot_base.narrow(0, spec.offset_bytes, spec.num_bytes).view(spec.view_shape)
            (weights if spec.kind == "packed_fp4" else scales)[spec.name] = region
        return weights, scales


def _align_up(value: int, alignment: int = SLOT_REGION_ALIGN_BYTES) -> int:
    if value % alignment == 0:
        return value
    return (value // alignment + 1) * alignment


@dataclass(frozen=True)
class _AdmissionPlan:
    """Pure result of the transactional planning phase (nothing applied yet).

    ``hits`` and ``admissions`` are each in request order; ``request_order``
    preserves the original interleaving so the pool can emit one slot id per
    requested expert, in request order.
    """

    hits: tuple[tuple[int, int], ...]  # (expert_key, resident slot)
    admissions: tuple[tuple[int, int], ...]  # (expert_key, fresh slot: reused-free or evicted)
    evictions: tuple[tuple[int, int], ...]  # (victim expert_key, freed slot)
    request_order: tuple[int, ...]
    free_slot_count: int  # admissions served by the free list (rest by evictions)

    def slot_for(self, key: int) -> int:
        """Slot id for a requested expert key (hits keep their resident slot)."""
        for candidate_key, slot in self.hits:
            if candidate_key == key:
                return slot
        for candidate_key, slot in self.admissions:
            if candidate_key == key:
                return slot
        raise KeyError(f"key {key} missing from admission plan")


class _SlotResidencyLedger:
    """Cache state machine (LRU + step locks) over abstract slot ids.

    Deliberately torch-free: state management is decoupled from tensor
    allocation (SRP); the pool owns the physical arena and delegates all
    residency decisions here.
    """

    def __init__(self, num_slots: int, num_experts: int):
        self.num_slots = num_slots
        self.num_experts = num_experts
        self._lru: OrderedDict[int, int] = OrderedDict()  # expert_key -> slot, oldest first
        self._slot_keys: list[int | None] = [None] * num_slots
        self._free_slots: list[int] = list(range(num_slots))
        self._lock_counts: list[int] = [0] * num_slots

    @staticmethod
    def expert_key(layer_idx: int, expert_id: int, num_experts: int) -> int:
        return layer_idx * num_experts + expert_id

    @staticmethod
    def split_key(key: int, num_experts: int) -> tuple[int, int]:
        return divmod(key, num_experts)

    def lookup(self, key: int) -> int | None:
        return self._lru.get(key)

    def resident_count(self) -> int:
        return len(self._lru)

    def free_count(self) -> int:
        return len(self._free_slots)

    def plan_admission(self, keys: Sequence[int]) -> _AdmissionPlan:
        """Compute the whole step's admission atomically or raise.

        A slot locked by the in-flight step (already in ``_lock_counts`` or a
        hit of this very plan) is never offered as an eviction victim, so a
        step's own top-k slots can never collide.
        """
        hits: list[tuple[int, int]] = []
        misses: list[int] = []
        for key in keys:
            slot = self._lru.get(key)
            if slot is None:
                misses.append(key)
            else:
                hits.append((key, slot))

        protected: set[int] = {slot for _, slot in hits}
        free_taken = self._free_slots[-len(misses) :] if misses else []
        evictions: list[tuple[int, int]] = []
        for _ in range(len(misses) - len(free_taken)):
            victim = self._find_evictable(protected)
            evictions.append(victim)
            protected.add(victim[1])
        admissions = tuple((key, slot) for key, slot in zip(misses, list(free_taken) + [slot for _, slot in evictions]))
        return _AdmissionPlan(
            hits=tuple(hits),
            admissions=admissions,
            evictions=tuple(evictions),
            request_order=tuple(keys),
            free_slot_count=len(free_taken),
        )

    def _find_evictable(self, protected: set[int]) -> tuple[int, int]:
        for key, slot in self._lru.items():  # OrderedDict: oldest first
            if slot not in protected and self._lock_counts[slot] == 0:
                return key, slot
        raise SlotExhaustionError(
            f"cannot admit expert: {self.resident_count()} resident slots and none evictable "
            f"({self.free_count()} free, {sum(1 for c in self._lock_counts if c)} locked)"
        )

    def commit(self, plan: _AdmissionPlan) -> None:
        for victim_key, victim_slot in plan.evictions:
            del self._lru[victim_key]
            self._slot_keys[victim_slot] = None
        if plan.free_slot_count:
            del self._free_slots[-plan.free_slot_count :]
        for key, slot in plan.admissions:
            self._lru[key] = slot  # inserts at MRU end
            self._slot_keys[slot] = key
        for key, _slot in plan.hits:
            self._lru.move_to_end(key)

    def lock_slots(self, slots: Sequence[int]) -> None:
        for slot in slots:
            self._lock_counts[slot] += 1

    def unlock_slots(self, slots: Sequence[int]) -> None:
        for slot in slots:
            if self._lock_counts[slot] <= 0:
                raise ValueError(f"slot {slot} unlocked too often")
            self._lock_counts[slot] -= 1


@runtime_checkable
class WeightProviderProtocol(Protocol):
    """Dependency boundary between slot residency and weight fetching (DIP).

    The pool never knows where expert weights come from (safetensors mmap,
    KV-to-DDR spiller, test mock); it only asks for the pinned host tensor of
    one expert parameter and performs the single in-place
    ``slot_view.copy_(pinned_cpu_weight, non_blocking=False)`` itself.
    Implementations must return pre-allocated, contiguous, CPU tensors of the
    exact shape ``ExpertTensorLayout.spec_for(param_key).view_shape`` --
    serving may not allocate.
    """

    def pinned_cpu_weight(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor: ...


@dataclass
class SlotPoolStats:
    """Live diagnostics counters (mutated in place by the pool)."""

    hits: int = 0
    loads: int = 0
    evictions: int = 0


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

    See the module docstring for the memory contracts. Device-side consumers
    read ``expert_slot_table[layer_idx]`` (int32, -1 = miss) directly on
    device; ``weight_views(slot_id)``/``scale_views(slot_id)`` are the
    pre-sliced, fixed-address compute operands.
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
        for slot_id in range(num_slots):
            weights, scales = self._layout.slice_slot_views(self._slot_arena, slot_id)
            self._weight_views.append(weights)
            self._scale_views.append(scales)

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
        untouched. Fills are strictly ``view.copy_(pinned, non_blocking=False)``.
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
        # 1. Fetch and validate every source buffer *before* mutating any state,
        #    so a misbehaving provider cannot leave the pool half-committed.
        pending_fills: list[tuple[int, int, list[torch.Tensor]]] = []  # (expert_id, slot, sources)
        for key, slot in plan.admissions:
            _admitted_layer, expert_id = _SlotResidencyLedger.split_key(key, self._config.num_routed_experts)
            sources = []
            for spec in self._layout.specs:
                source = host_pinned_storage.pinned_cpu_weight(layer_idx, expert_id, spec.param_key)
                self._assert_transfer_compatible(spec.param_key, source)
                sources.append(source)
            pending_fills.append((expert_id, slot, sources))

        # 2. Bookkeeping (ledger + device table), then bytes. Copies cannot fail
        #    here: shapes/dtypes were validated above.
        self._ledger.commit(plan)
        for victim_key, _slot in plan.evictions:
            victim_layer, victim_expert = _SlotResidencyLedger.split_key(victim_key, self._config.num_routed_experts)
            self._expert_slot_table[victim_layer, victim_expert] = UNRESIDENT_SLOT_ID
            self.stats.evictions += 1
        for key, slot in plan.hits:
            _hit_layer, expert_id = _SlotResidencyLedger.split_key(key, self._config.num_routed_experts)
            self._expert_slot_table[layer_idx, expert_id] = slot
            self.stats.hits += 1
        for expert_id, slot, sources in pending_fills:
            self._expert_slot_table[layer_idx, expert_id] = slot
            for spec, source in zip(self._layout.specs, sources):
                destination = (self._weight_views if spec.kind == "packed_fp4" else self._scale_views)[slot][spec.name]
                destination.copy_(source, non_blocking=False)
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


@runtime_checkable
class RouteResolverProtocol(Protocol):
    """Per-layer routing policy: fills ``out`` in place (hash vs score gating)."""

    def resolve(self, layer_idx: int, network_input: torch.Tensor, out: torch.Tensor) -> None: ...


class HashRouteResolver:
    """Deterministic ``tid2eid`` routing for layers ``[0, num_hash_layers)``.

    Matches ``Gate.forward`` in the reference implementation: expert indices
    are a pure function of the token id -- ``indices = tid2eid[input_ids]`` --
    and each hash layer owns its own table (the checkpoint ships one
    ``layers.<i>.ffn.gate.tid2eid`` per hash layer). Accepts either a shared
    ``[vocab, top_k]`` table or the full per-layer ``[num_hash_layers, vocab,
    top_k]`` stack. Allocation-free: ``torch.index_select(..., out=)`` writes
    caller storage. ``out`` must be ``int32 [num_tokens, top_k]``;
    ``network_input`` is the ``int64`` token-id vector.
    """

    def __init__(self, config: DeepSeekV4MoEConfig, tid2eid: torch.Tensor):
        shared_shape = (config.vocab_size, config.top_k)
        layered_shape = (config.num_hash_layers, config.vocab_size, config.top_k)
        if tuple(tid2eid.shape) not in (shared_shape, layered_shape):
            raise ValueError(
                f"tid2eid must be {shared_shape} (shared) or {layered_shape} (per-layer), got {tuple(tid2eid.shape)}"
            )
        if tid2eid.dtype != torch.int32:
            raise ValueError(f"tid2eid must be int32 to mirror the checkpoint, got {tid2eid.dtype}")
        if int(tid2eid.max()) >= config.num_routed_experts:
            raise ValueError("tid2eid references an expert outside [0, num_routed_experts)")
        self._config = config
        self._tid2eid = tid2eid

    def resolve(self, layer_idx: int, network_input: torch.Tensor, out: torch.Tensor) -> None:
        if layer_idx >= self._config.num_hash_layers:
            raise ValueError(f"layer {layer_idx} is score-gated (hash layers are [0, {self._config.num_hash_layers}))")
        if network_input.dtype != torch.int64:
            raise ValueError(f"token ids must be int64, got {network_input.dtype}")
        table = self._tid2eid if self._tid2eid.ndim == 2 else self._tid2eid[layer_idx]
        torch.index_select(table, 0, network_input, out=out)


class ScoreRouteResolver:
    """Bias-shifted top-k gating for layers ``[num_hash_layers, num_layers)``.

    Matches the reference ``noaux_tc`` behaviour: the routing bias shifts
    *selection only*; routing weights are gathered from the original scores
    (compute-side concern, not handled here).

    Pilot note: ``torch.topk`` cannot write into caller storage; the two small
    result tensors it returns are the documented exception to the zero-alloc
    contract. The 950PR path replaces this with an aclnn V5 top-k plan writing
    pre-allocated outputs. ``out`` must be ``int64 [num_tokens, top_k]``.
    """

    def __init__(
        self,
        config: DeepSeekV4MoEConfig,
        gate_bias: torch.Tensor,
        max_batch_tokens: int,
    ):
        expected_bias_shape = (config.num_layers, config.num_routed_experts)
        if tuple(gate_bias.shape) not in (expected_bias_shape, (config.num_routed_experts,)):
            raise ValueError(
                f"gate_bias must be {expected_bias_shape} or ({config.num_routed_experts},), "
                f"got {tuple(gate_bias.shape)}"
            )
        if gate_bias.dtype != torch.float32:
            raise ValueError(f"gate_bias must be float32, got {gate_bias.dtype}")
        self._config = config
        self._gate_bias = gate_bias
        self._max_batch_tokens = max_batch_tokens
        self._combined_scores = torch.empty(max_batch_tokens, config.num_routed_experts, dtype=torch.float32)

    def resolve(self, layer_idx: int, network_input: torch.Tensor, out: torch.Tensor) -> None:
        if layer_idx < self._config.num_hash_layers:
            raise ValueError(
                f"layer {layer_idx} is hash-routed; score gating starts at layer {self._config.num_hash_layers}"
            )
        num_tokens, num_experts = network_input.shape
        if num_experts != self._config.num_routed_experts:
            raise ValueError(f"scores must cover {self._config.num_routed_experts} experts, got {num_experts}")
        if num_tokens > self._max_batch_tokens:
            raise ValueError(f"{num_tokens} tokens exceed pre-located buffer of {self._max_batch_tokens}")
        bias_row = self._gate_bias if self._gate_bias.ndim == 1 else self._gate_bias[layer_idx]
        combined = self._combined_scores[:num_tokens]  # pre-allocated scratch
        combined.copy_(network_input)
        combined += bias_row
        _values, indices = torch.topk(combined, self._config.top_k, dim=-1)
        out.copy_(indices)
