"""Cache state machine (LRU + step locks) over abstract slot ids.

Deliberately torch-free (SRP): state management is decoupled from tensor
allocation; the slot pool owns the physical arena and delegates all residency
decisions here.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass


class SlotExhaustionError(RuntimeError):
    """Raised when a step cannot lock all of its top-k slots without evicting a locked slot."""


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
    """LRU residency + step-lock bookkeeping.

    Locking contract: a slot locked by the in-flight step is never offered as
    an eviction victim, so a step's own top-k slots can never collide.
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
        """Compute the whole step's admission atomically or raise."""
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
