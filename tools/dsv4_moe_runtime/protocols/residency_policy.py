"""Residency policy dependency boundary (DIP): pluggable eviction strategies.

``StaticExpertSlotPool`` owns only physical facts (arena slicing, slot ids,
locks, DMA fills); *which* resident expert to evict and how priorities are
computed belong entirely to the injected :class:`EvictionPolicyProtocol`.

Interface notes (extending the mandated three-method surface, call-compatible):

* ``plan_admissions`` accepts an optional ``locked_slots`` keyword: the pool's
  physical exclusivity set (slots held by in-flight steps). Policies must not
  plan those slots as eviction victims; the default ``()`` keeps the mandated
  three-argument call shape valid.
* :meth:`EvictionPolicyProtocol.commit_decision` applies a decision to the
  policy's internal bookkeeping *after* the pool has validated the provider
  sources, preserving the transactional contract (a provider failure before
  commit leaves both pool and policy untouched).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class SlotExhaustionError(RuntimeError):
    """Raised when a step cannot lock all of its top-k slots without evicting a locked slot."""


@dataclass(frozen=True)
class AdmissionDecision:
    """Pure result of the transactional planning phase (nothing applied yet).

    ``hits`` and ``admissions`` are each in request order; ``request_order``
    preserves the original interleaving of the requested ``(layer, expert)``
    pairs so the pool can emit one slot id per requested expert.
    """

    hits: tuple[tuple[tuple[int, int], int], ...]  # ((layer, expert), resident slot)
    admissions: tuple[tuple[tuple[int, int], int], ...]  # ((layer, expert), fresh slot)
    evictions: tuple[tuple[tuple[int, int], int], ...]  # ((layer, expert) victim, freed slot)
    request_order: tuple[tuple[int, int], ...]
    free_slot_count: int  # admissions served by the free list (rest by evictions)

    def slot_for(self, key: tuple[int, int]) -> int:
        """Slot id for a requested ``(layer, expert)`` pair (hits keep their slot)."""
        for candidate_key, slot in self.hits:
            if candidate_key == key:
                return slot
        for candidate_key, slot in self.admissions:
            if candidate_key == key:
                return slot
        raise KeyError(f"key {key} missing from admission plan")


@runtime_checkable
class EvictionPolicyProtocol(Protocol):
    """Strategy interface for residency and eviction decisions (OCP / DIP).

    Implementations are pure planners: they never touch tensors. The pool
    feeds them physical facts and applies the returned
    :class:`AdmissionDecision` to the hardware.
    """

    def register_access(self, layer_idx: int, expert_id: int, token_idx: int) -> None:
        """Record one access of an expert during ``token_idx`` (frequency/recency)."""

    def advance_generation(self, completed_token_idx: int) -> None:
        """Close out ``completed_token_idx``; age every not-just-accessed expert."""

    def plan_admissions(
        self,
        requested: Sequence[tuple[int, int]],
        resident_slots: Mapping[tuple[int, int], int],
        free_slots: Sequence[int],
        locked_slots: Sequence[int] = (),
    ) -> AdmissionDecision:
        """Plan the whole step atomically, or raise ``SlotExhaustionError``."""

    def commit_decision(self, decision: AdmissionDecision) -> None:
        """Apply a (already pool-validated) decision to the policy's bookkeeping."""
