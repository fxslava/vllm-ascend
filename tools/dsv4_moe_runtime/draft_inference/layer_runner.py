"""One transformer layer's routed-expert forward over the static slot pool.

Per-layer flow (single token, top-k routed experts):

1. route -- hash layers 0-2 through ``HashRouteResolver`` (``tid2eid``),
   layers 3+ through ``ScoreRouteResolver`` (bias-shifted top-k);
2. acquire -- ``StaticExpertSlotPool.acquire_for_step`` locks and refreshes
   the k slots (streamed DMA fills from the host-pinned provider);
3. project -- per expert: one ``ExpertKernelRunner.execute_expert`` call
   handing the slot's FP4 weight/scale views plus the scratchpad rows
   (gate/up projections, in-place SwiGLU, down projection) to the kernel
   seam -- today the safe surrogate in ``hardware/dummy_kernel.py``, later
   the native vllm-ascend kernel with the identical signature;
4. accumulate -- routing-weighted sum into the fp32 accumulator, all in place.
"""

from __future__ import annotations

from dsv4_moe_runtime.core.config import DeepSeekV4MoEConfig
from dsv4_moe_runtime.core.layout import ExpertTensorLayout
from dsv4_moe_runtime.core.slot_pool import StaticExpertSlotPool
from dsv4_moe_runtime.draft_inference.scratchpad import DecodeScratchpad
from dsv4_moe_runtime.hardware.dummy_kernel import ExpertKernelRunner
from dsv4_moe_runtime.protocols.provider import WeightProviderProtocol
from dsv4_moe_runtime.routing.hash_router import HashRouteResolver
from dsv4_moe_runtime.routing.score_router import ScoreRouteResolver


class MoELayerRunner:
    """Executes one MoE block against the shared scratchpad (single-flight)."""

    def __init__(
        self,
        config: DeepSeekV4MoEConfig,
        layout: ExpertTensorLayout,
        pool: StaticExpertSlotPool,
        scratchpad: DecodeScratchpad,
        expert_kernel: ExpertKernelRunner,
        hash_resolver: HashRouteResolver,
        score_resolver: ScoreRouteResolver,
    ):
        self._config = config
        self._layout = layout
        self._pool = pool
        self._scratchpad = scratchpad
        self._expert_kernel = expert_kernel
        self._hash = hash_resolver
        self._score = score_resolver

    def run_layer(self, layer_idx: int, provider: WeightProviderProtocol) -> tuple[int, int]:
        """Advance one MoE layer; returns (hits, misses) of the slot acquisition."""
        expert_ids = self._resolve_expert_ids(layer_idx)
        reservation = self._pool.acquire_for_step(layer_idx, expert_ids, provider)
        try:
            self._project_experts(layer_idx, expert_ids, reservation.slot_ids)
        finally:
            self._pool.release_step(reservation)
        misses = sum(1 for expert_id in expert_ids if self._pool.slot_of(layer_idx, expert_id) >= 0)
        hits = len(expert_ids) - misses
        return hits, misses

    # -------------------------------------------------------------- internals

    def _resolve_expert_ids(self, layer_idx: int) -> list[int]:
        token_ids = self._scratchpad["token_ids"]
        route_out = self._scratchpad["route_experts"]
        if layer_idx < self._config.num_hash_layers:
            self._hash.resolve(layer_idx, token_ids, route_out)
        else:
            # ScoreRouteResolver copies its int64 top-k indices in place; the
            # copy_ casts them into the int32 route buffer.
            self._score.resolve(layer_idx, self._scratchpad["route_scores"], route_out)
        return [int(value) for value in route_out[0].tolist()]

    def _project_experts(self, layer_idx: int, expert_ids: list[int], slot_ids: tuple[int, ...]) -> None:
        gathered_x = self._scratchpad["gathered_x"]
        routed_accum = self._scratchpad["routed_accum"]
        routed_accum.zero_()
        self._normalize_routing_weights(layer_idx, expert_ids)
        weights = self._scratchpad["route_weights"]
        for position, slot_id in enumerate(slot_ids):
            gathered_x[position].copy_(self._scratchpad["hidden"][0])
            self._project_one_expert(position, slot_id)
            self._accumulate_expert(position, float(weights[position]))

    def _normalize_routing_weights(self, layer_idx: int, expert_ids: list[int]) -> None:
        """Hash layers weight uniformly; score layers normalize their gate scores."""
        weights = self._scratchpad["route_weights"]
        if layer_idx < self._config.num_hash_layers:
            weights.fill_(1.0 / self._config.top_k)
            return
        scores = self._scratchpad["route_scores"]
        for position, expert_id in enumerate(expert_ids):
            weights[position] = scores[0, expert_id]
        weights.div_(weights.sum())

    def _project_one_expert(self, position: int, slot_id: int) -> None:
        """One fused expert projection through the kernel seam, all in place.

        Hands the kernel exactly the operands the native vllm-ascend kernel
        will receive: the hidden row, the slot's packed FP4 weights and E8M0
        scales, and the four scratchpad rows (gate/up/activated/down).
        """
        weight_views = self._pool.weight_views(slot_id)
        scale_views = self._pool.scale_views(slot_id)
        self._expert_kernel.execute_expert(
            x=self._scratchpad["gathered_x"][position : position + 1],
            w1=weight_views["w1"],
            w2=weight_views["w2"],
            w3=weight_views["w3"],
            w1_scale=scale_views["w1"],
            w2_scale=scale_views["w2"],
            w3_scale=scale_views["w3"],
            gate_out=self._scratchpad["gate_out"][position : position + 1],
            up_out=self._scratchpad["up_out"][position : position + 1],
            activated=self._scratchpad["activated"][position : position + 1],
            down_out=self._scratchpad["down_out"][position : position + 1],
            label=f"slot{slot_id}",
        )

    def _accumulate_expert(self, position: int, weight: float) -> None:
        down_row = self._scratchpad["down_row_f32"][position]
        down_row.copy_(self._scratchpad["down_out"][position])  # bf16 -> fp32, in place
        down_row.mul_(weight)
        self._scratchpad["routed_accum"][0].add_(down_row)
