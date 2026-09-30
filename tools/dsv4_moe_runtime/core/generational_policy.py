"""Generational radix-sorted eviction policy (production default).

Eviction model over a generational timeline (relative to the most recently
*completed* token):

* **Gen 0** -- accessed during that token; the protected core.
* **Gen 1** -- accessed the token before, inactive since.
* **Gen 2+ (cold)** -- inactive for >= 2 tokens (generation saturates at 2).

Deterministic hash-routed layers (``tid2eid``, layers 0-2) receive *soft
immunity*: while in Gen 0 they rank as the last possible eviction victims
("pinned"), so late-layer churn can never displace the hash core unless no
other resident exists (which would otherwise deadlock small pools).

Radix ranking (compact 64-bit math, vectorized over CPU tensors):

* ``ranking_key`` follows the mandated composite
  ``(Priority << 48) | (Generation << 32) | (Frequency << 16) | Recency``
  with Priority 3 = pinned (hash + Gen0), 2 = hash, 1 = Gen0, 0 = cold.
* ``evict_order_key`` is the ascending sort key whose first candidates are
  the victims: bits 56-57 = class (Gen>=1 -> 0, Gen0 -> 1, pinned hash-Gen0
  -> 2, so the protected core sorts last but stays reachable as a last
  resort -- immunity is a rank, never a deadlock), bits 48-55 = inverted
  generation (coldest generation first), bits 24-47 = frequency ascending
  (coldest first), bits 0-23 = inverted age (oldest first). Non-candidates
  (not resident / requested / locked) receive the sentinel maximum.

Allocation contract: the plan-sized buffers (candidate keys, sort values and
indices, masks) are allocated once in ``__init__`` and reused in place --
notably ``torch.sort(..., out=)``. Remaining temporaries are constant-shape
host tensors from torch's caching CPU allocator; no device memory is ever
allocated during planning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from ..protocols.residency_policy import AdmissionDecision, SlotExhaustionError

GENERATION_COLD_CAP = 2
FREQUENCY_FIELD_MAX = (1 << 16) - 1
AGE_FIELD_MAX = (1 << 24) - 1
CLASS_SHIFT = 56
GEN_FIELD_SHIFT = 48
FREQ_FIELD_SHIFT = 24
EVICT_KEY_SENTINEL = (1 << 62) - 1
PRIORITY_PINNED = 3
PRIORITY_HASH = 2
PRIORITY_GEN0 = 1
PRIORITY_COLD = 0


class GenerationalRadixPolicy:
    """Production eviction strategy: generational aging + radix-sorted victims."""

    def __init__(self, config, num_slots: int, device: str = "cpu"):
        num_keys = config.num_layers * config.num_routed_experts
        self._num_experts = config.num_routed_experts
        self._num_slots = num_slots
        self._current_token = 0
        self._last_completed = -1

        layer_is_hash = torch.arange(config.num_layers) < config.num_hash_layers
        self._hash_mask = layer_is_hash.repeat_interleave(config.num_routed_experts)
        self._generation = torch.full((num_keys,), GENERATION_COLD_CAP, dtype=torch.int64)
        self._frequency = torch.zeros(num_keys, dtype=torch.int64)
        self._last_token = torch.full((num_keys,), -1, dtype=torch.int64)
        self._resident = torch.zeros(num_keys, dtype=torch.bool)
        self._resident_slot = torch.full((num_keys,), -1, dtype=torch.int64)
        # AOT planning scratch, reused in place on every call.
        self._requested_mask = torch.zeros(num_keys, dtype=torch.bool)
        self._key_locked = torch.zeros(num_keys, dtype=torch.bool)
        self._locked_slot_mask = torch.zeros(num_slots, dtype=torch.bool)
        self._slot_clamped = torch.zeros(num_keys, dtype=torch.int64)
        self._evict_keys = torch.zeros(num_keys, dtype=torch.int64)
        self._sort_values = torch.zeros(num_keys, dtype=torch.int64)
        self._sort_indices = torch.zeros(num_keys, dtype=torch.int64)
        self._requested_indices = torch.zeros(config.top_k, dtype=torch.int64)

    # ---------------------------------------------------- introspection (tests)

    def _key_index(self, layer_idx: int, expert_id: int) -> int:
        return layer_idx * self._num_experts + expert_id

    def generation_of(self, layer_idx: int, expert_id: int) -> int:
        return int(self._generation[self._key_index(layer_idx, expert_id)])

    def frequency_of(self, layer_idx: int, expert_id: int) -> int:
        return int(self._frequency[self._key_index(layer_idx, expert_id)])

    def ranking_key(self, layer_idx: int, expert_id: int) -> int:
        """Mandated composite ranking key (higher = more protected)."""
        index = self._key_index(layer_idx, expert_id)
        generation = int(self._generation[index])
        is_hash = bool(self._hash_mask[index])
        if is_hash and generation == 0:
            priority = PRIORITY_PINNED
        elif is_hash:
            priority = PRIORITY_HASH
        elif generation == 0:
            priority = PRIORITY_GEN0
        else:
            priority = PRIORITY_COLD
        frequency = int(self._frequency[index]) & FREQUENCY_FIELD_MAX
        recency = int(self._last_token[index]) & FREQUENCY_FIELD_MAX
        return (priority << 48) | (generation << 32) | (frequency << 16) | recency

    def evict_order_key(self, layer_idx: int, expert_id: int) -> int:
        """Debug accessor: ascending sort key (smallest = evicted first)."""
        index = self._key_index(layer_idx, expert_id)
        self._fill_evict_keys(self._evict_keys)
        return int(self._evict_keys[index])

    # -------------------------------------------------- EvictionPolicyProtocol

    def register_access(self, layer_idx: int, expert_id: int, token_idx: int) -> None:
        index = self._key_index(layer_idx, expert_id)
        self._generation[index] = 0
        self._frequency[index] = min(int(self._frequency[index]) + 1, FREQUENCY_FIELD_MAX)
        self._last_token[index] = token_idx

    def advance_generation(self, completed_token_idx: int) -> None:
        if completed_token_idx <= self._last_completed:
            return  # idempotent per token boundary
        self._last_completed = completed_token_idx
        self._current_token = completed_token_idx + 1
        aging = self._last_token < completed_token_idx
        just_completed = ~aging & (self._last_token >= 0)
        self._generation[aging] = (self._generation[aging] + 1).clamp(max=GENERATION_COLD_CAP)
        self._generation[just_completed] = 0  # accessed in the completed token -> Gen 0
        self._frequency[aging] >>= 1  # decay: cold cores cool down further

    def plan_admissions(
        self,
        requested: Sequence[tuple[int, int]],
        resident_slots: Mapping[tuple[int, int], int],
        free_slots: Sequence[int],
        locked_slots: Sequence[int] = (),
    ) -> AdmissionDecision:
        """Plan the whole step atomically, or raise ``SlotExhaustionError``."""
        hits: list[tuple[tuple[int, int], int]] = []
        misses: list[tuple[int, int]] = []
        for pair in requested:
            slot = resident_slots.get(pair)
            if slot is None:
                misses.append(pair)
            else:
                hits.append((pair, slot))
        self._build_candidate_keys(requested, locked_slots)

        free_taken = list(free_slots[-len(misses) :]) if misses else []
        victim_count = len(misses) - len(free_taken)
        evictions = self._select_victims(victim_count)
        admissions = tuple((pair, slot) for pair, slot in zip(misses, free_taken + [slot for _, slot in evictions]))
        return AdmissionDecision(
            hits=tuple(hits),
            admissions=admissions,
            evictions=tuple(evictions),
            request_order=tuple(requested),
            free_slot_count=len(free_taken),
        )

    def commit_decision(self, decision: AdmissionDecision) -> None:
        """Mirror the pool-applied decision into the policy's residency state."""
        for victim_key, _slot in decision.evictions:
            index = self._key_index(victim_key[0], victim_key[1])
            self._resident[index] = False
            self._resident_slot[index] = -1
        for key, slot in decision.admissions:
            index = self._key_index(key[0], key[1])
            self._resident[index] = True
            self._resident_slot[index] = slot

    # -------------------------------------------------------------- internals

    def _fill_evict_keys(self, out: torch.Tensor) -> None:
        """Vectorized ``evict_order_key`` for every expert key (ascending = first)."""
        is_gen0 = (self._generation == 0).to(torch.int64)
        pinned = (is_gen0 * self._hash_mask.to(torch.int64)).to(torch.int64)
        gen_capped = self._generation.clamp(max=GENERATION_COLD_CAP)
        inv_gen = (GENERATION_COLD_CAP - gen_capped) * (1 - is_gen0)
        age = (self._current_token - self._last_token).clamp(0, AGE_FIELD_MAX)
        out.copy_(
            ((is_gen0 + pinned) << CLASS_SHIFT)
            + (inv_gen << GEN_FIELD_SHIFT)
            + (self._frequency.clamp(max=FREQUENCY_FIELD_MAX) << FREQ_FIELD_SHIFT)
            + (AGE_FIELD_MAX - age)
        )

    def _build_candidate_keys(self, requested: Sequence[tuple[int, int]], locked_slots: Sequence[int]) -> None:
        """Refresh the AOT candidate sort keys for one planning call."""
        for position, (layer_idx, expert_id) in enumerate(requested):
            self._requested_indices[position] = self._key_index(layer_idx, expert_id)
        self._requested_mask.zero_()
        self._requested_mask[self._requested_indices[: len(requested)]] = True
        self._locked_slot_mask.zero_()
        for slot in locked_slots:
            self._locked_slot_mask[slot] = True
        self._slot_clamped.copy_(self._resident_slot.clamp(min=0))
        torch.index_select(self._locked_slot_mask, 0, self._slot_clamped, out=self._key_locked)
        self._fill_evict_keys(self._evict_keys)
        ineligible = ~self._resident | self._requested_mask | self._key_locked
        self._evict_keys[ineligible] = EVICT_KEY_SENTINEL

    def _select_victims(self, victim_count: int) -> list[tuple[tuple[int, int], int]]:
        """Pick the ``victim_count`` coldest residents in radix order."""
        if victim_count <= 0:
            return []
        torch.sort(self._evict_keys, stable=True, out=(self._sort_values, self._sort_indices))
        victims: list[tuple[tuple[int, int], int]] = []
        for position in range(victim_count):
            if int(self._sort_values[position]) >= EVICT_KEY_SENTINEL:
                break
            key_index = int(self._sort_indices[position])
            victims.append((divmod(key_index, self._num_experts), int(self._resident_slot[key_index])))
        if len(victims) < victim_count:
            raise SlotExhaustionError(
                f"cannot admit expert: {int(self._resident.sum())} resident slots and none "
                "evictable (all remaining candidates are pinned, requested or locked)"
            )
        return victims
