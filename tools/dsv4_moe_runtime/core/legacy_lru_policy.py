"""Legacy flat-LRU eviction policy (backward-compatibility baseline).

This module migrates the original ``_SlotResidencyLedger`` state machine
unchanged behind the pluggable :class:`EvictionPolicyProtocol`, preserving
100% of the historical eviction semantics for existing baselines: free-slot
reuse from the top of the free list, strictly oldest-first LRU victims, and
identical exhaustion messages. New development should target
:class:`~tools.dsv4_moe_runtime.core.generational_policy.GenerationalRadixPolicy`.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence

from ..protocols.residency_policy import AdmissionDecision, SlotExhaustionError


class _SlotResidencyLedger:
    """LRU residency bookkeeping over packed integer expert keys.

    Migrated verbatim from the former ``core/ledger.py``. Physical step locks
    live in the pool; planning receives them as ``locked_slots``.
    """

    def __init__(self, num_slots: int, num_experts: int):
        self.num_slots = num_slots
        self.num_experts = num_experts
        self._lru: OrderedDict[int, int] = OrderedDict()  # expert_key -> slot, oldest first
        self._free_slots: list[int] = list(range(num_slots))

    @staticmethod
    def expert_key(layer_idx: int, expert_id: int, num_experts: int) -> int:
        return layer_idx * num_experts + expert_id

    def touch(self, key: int) -> None:
        if key in self._lru:
            self._lru.move_to_end(key)

    def plan_admission(self, keys: Sequence[int], locked_slots: Sequence[int] = ()) -> AdmissionDecision:
        """Compute the whole step's admission atomically or raise."""
        locked_set = frozenset(locked_slots)
        hits: list[tuple[tuple[int, int], int]] = []
        misses: list[int] = []
        for key in keys:
            slot = self._lru.get(key)
            if slot is None:
                misses.append(key)
            else:
                hits.append((divmod(key, self.num_experts), slot))

        protected: set[int] = {slot for _, slot in hits} | locked_set
        free_taken = self._free_slots[-len(misses) :] if misses else []
        evictions: list[tuple[tuple[int, int], int]] = []
        for _ in range(len(misses) - len(free_taken)):
            victim = self._find_evictable(protected, len(locked_set))
            evictions.append(victim)
            protected.add(victim[1])
        admissions = tuple(
            (divmod(key, self.num_experts), slot)
            for key, slot in zip(misses, list(free_taken) + [slot for _, slot in evictions])
        )
        return AdmissionDecision(
            hits=tuple(hits),
            admissions=admissions,
            evictions=tuple(evictions),
            request_order=tuple(divmod(key, self.num_experts) for key in keys),
            free_slot_count=len(free_taken),
        )

    def _find_evictable(self, protected: set[int], locked_count: int) -> tuple[tuple[int, int], int]:
        for key, slot in self._lru.items():  # OrderedDict: oldest first
            if slot not in protected:
                return divmod(key, self.num_experts), slot
        raise SlotExhaustionError(
            f"cannot admit expert: {len(self._lru)} resident slots and none evictable "
            f"({len(self._free_slots)} free, {locked_count} locked)"
        )

    def commit(self, decision: AdmissionDecision) -> None:
        num_experts = self.num_experts
        for victim_key, _slot in decision.evictions:
            del self._lru[victim_key[0] * num_experts + victim_key[1]]
        if decision.free_slot_count:
            del self._free_slots[-decision.free_slot_count :]
        for key, slot in decision.admissions:
            self._lru[key[0] * num_experts + key[1]] = slot  # inserts at MRU end
        for key, _slot in decision.hits:
            self._lru.move_to_end(key[0] * num_experts + key[1])


class LegacyLruPolicy:
    """Flat-LRU strategy (the pre-refactor default) as an ``EvictionPolicyProtocol``.

    Same-technique bookkeeping as ``_SlotResidencyLedger``: access registration
    is an LRU touch, generational advance is a no-op, and eviction victims are
    strictly the oldest residents. Free slots are consumed from the top of the
    pool's free list, exactly like the historical behaviour.
    """

    def __init__(self, config, num_slots: int):
        self._ledger = _SlotResidencyLedger(num_slots, config.num_routed_experts)

    @property
    def ledger(self) -> _SlotResidencyLedger:
        """Direct ledger access for baseline tests that introspect LRU order."""
        return self._ledger

    def register_access(self, layer_idx: int, expert_id: int, token_idx: int) -> None:
        del token_idx
        self._ledger.touch(self._ledger.expert_key(layer_idx, expert_id, self._ledger.num_experts))

    def advance_generation(self, completed_token_idx: int) -> None:
        del completed_token_idx  # flat LRU has no generations

    def plan_admissions(
        self,
        requested: Sequence[tuple[int, int]],
        resident_slots: Mapping[tuple[int, int], int],
        free_slots: Sequence[int],
        locked_slots: Sequence[int] = (),
    ) -> AdmissionDecision:
        del resident_slots, free_slots  # the ledger carries the authoritative LRU state
        keys = [
            self._ledger.expert_key(layer_idx, expert_id, self._ledger.num_experts)
            for layer_idx, expert_id in requested
        ]
        return self._ledger.plan_admission(keys, locked_slots=locked_slots)

    def commit_decision(self, decision: AdmissionDecision) -> None:
        self._ledger.commit(decision)
