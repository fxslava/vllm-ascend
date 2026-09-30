"""Synthetic MoE router trace generation: mixture-Zipf activations.

The whole decode trace is pre-generated ahead of any timed loop (AOT): the
measured section of a benchmark never draws random numbers.
"""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass

from ..core.config import DeepSeekV4MoEConfig

_SEED_SALT_UNIVERSE = 0x5EED_0001
_SEED_SALT_TRACE = 0xA11CE


@dataclass(frozen=True)
class TraceStep:
    layer_idx: int
    expert_ids: tuple[int, ...]
    routed_by: str  # "hash" (layers 0-2) or "score" (layers 3+)


class RouterTraceSimulator:
    """Pre-generates the full decode trace ahead of the timed loop.

    Activation model: ``hot_ratio`` of each step's top-k draws come from a
    small recurring hot head, the rest from a long cold tail; both buckets are
    sampled from truncated Zipf weights over popularity ranks (ranks are a
    seeded permutation of expert ids, so popularity is not coupled to id).
    Layers ``[0, num_hash_layers)`` resolve through per-layer ``tid2eid``-style
    tables (expert ids are a pure function of the token id, exactly like the
    checkpoint's hash layers); layers 3+ emulate score top-k output by sampling
    the same mixture (deduplicated to exactly ``top_k`` unique experts).
    """

    def __init__(
        self,
        model_config: DeepSeekV4MoEConfig,
        num_experts_available: int,
        hot_experts: int,
        hot_ratio: float,
        zipf_exponent: float,
        seed: int,
        num_layers: int | None = None,
        num_hash_layers: int | None = None,
        first_layer_idx: int = 0,
    ):
        num_layers = model_config.num_layers if num_layers is None else num_layers
        num_hash_layers = model_config.num_hash_layers if num_hash_layers is None else num_hash_layers
        if first_layer_idx < 0:
            raise ValueError(f"first_layer_idx must be >= 0, got {first_layer_idx}")
        if not 1 <= hot_experts < num_experts_available:
            raise ValueError(f"hot_experts {hot_experts} outside [1, {num_experts_available})")
        if not 0.0 <= hot_ratio <= 1.0:
            raise ValueError(f"hot_ratio {hot_ratio} outside [0, 1]")
        if num_hash_layers > num_layers:
            raise ValueError(f"{num_hash_layers} hash layers exceed {num_layers} traced layers")
        self._model_config = model_config
        self._num_layers = num_layers
        self._num_hash_layers = num_hash_layers
        self._first_layer_idx = first_layer_idx
        self._hot_ratio = hot_ratio

        rng = random.Random(seed ^ _SEED_SALT_UNIVERSE)
        universe = list(range(num_experts_available))
        rng.shuffle(universe)  # popularity rank order, decoupled from expert id
        self._hot_ids = universe[:hot_experts]
        self._cold_ids = universe[hot_experts:]
        self._hot_cdf = self._cumulative([self._rank_weight(rank, zipf_exponent) for rank in range(hot_experts)])
        self._cold_cdf = self._cumulative(
            [self._rank_weight(rank, zipf_exponent) for rank in range(hot_experts, num_experts_available)]
        )
        self._seed = seed

    @staticmethod
    def _rank_weight(rank: int, zipf_exponent: float) -> float:
        return 1.0 / (rank + 1) ** zipf_exponent

    @property
    def hot_expert_ids(self) -> tuple[int, ...]:
        """The recurring hot head (popularity ranks [0, hot_experts))."""
        return tuple(self._hot_ids)

    @staticmethod
    def _cumulative(weights: Sequence[float]) -> tuple[float, ...]:
        total = sum(weights)
        running = 0.0
        cdf: list[float] = []
        for weight in weights:
            running += weight / total
            cdf.append(running)
        return tuple(cdf)

    def _draw(self, ids: Sequence[int], cdf: Sequence[float], rng: random.Random) -> int:
        return ids[min(bisect_right(cdf, rng.random()), len(ids) - 1)]

    def sample_step(self, rng: random.Random) -> tuple[int, ...]:
        """One deduplicated top-k activation set, ordered most-popular first."""
        chosen: list[int] = []
        while len(chosen) < self._model_config.top_k:
            from_hot = rng.random() < self._hot_ratio
            if from_hot:
                expert = self._draw(self._hot_ids, self._hot_cdf, rng)
            else:
                expert = self._draw(self._cold_ids, self._cold_cdf, rng)
            if expert not in chosen:
                chosen.append(expert)
        return tuple(sorted(chosen, key=self._popularity_rank))

    def _popularity_rank(self, expert_id: int) -> int:
        if expert_id in self._hot_ids:
            return self._hot_ids.index(expert_id)
        return len(self._hot_ids) + self._cold_ids.index(expert_id)

    def generate_trace(self, num_steps: int) -> list[TraceStep]:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        rng = random.Random(self._seed ^ _SEED_SALT_TRACE)
        token_count = math.ceil(num_steps / self._num_layers)
        # tid2eid emulation: row per token, one independent table per hash layer.
        hash_tables: dict[int, list[tuple[int, ...]]] = {
            layer: [tuple(sorted(self.sample_step(rng))) for _ in range(token_count)]
            for layer in range(self._num_hash_layers)
        }
        trace: list[TraceStep] = []
        for step in range(num_steps):
            # Layer ids are absolute model indices: a dense prefix (DeepSeek-V2-Lite
            # replaces layer 0 with a plain MLP) shifts the traced window, while the
            # hash-routed layers stay the absolute [0, num_hash_layers) window the
            # residency policy pins.
            layer_idx = self._first_layer_idx + step % self._num_layers
            token = step // self._num_layers
            if layer_idx < self._num_hash_layers:
                trace.append(TraceStep(layer_idx, hash_tables[layer_idx][token], "hash"))
            else:
                trace.append(TraceStep(layer_idx, self.sample_step(rng), "score"))
        return trace
