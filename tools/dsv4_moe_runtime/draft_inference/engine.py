"""Draft single-step decode engine: slot pool + routing + V5 GEMMs + scratchpad.

Zero-allocation contract: ``DraftInferenceEngine.__init__`` allocates every
device tensor (embedding table, routing tables, slot pool arena, scratchpad);
:meth:`step` only writes in place, so a decode step changes neither
``memory_allocated()`` nor any buffer's ``data_ptr``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dsv4_moe_runtime.core.config import DeepSeekV4MoEConfig
from dsv4_moe_runtime.core.layout import ExpertTensorLayout
from dsv4_moe_runtime.core.slot_pool import StaticExpertSlotPool
from dsv4_moe_runtime.draft_inference.backends import DraftBackend
from dsv4_moe_runtime.draft_inference.layer_runner import MoELayerRunner
from dsv4_moe_runtime.draft_inference.scratchpad import DecodeScratchpad, ScratchpadShapes
from dsv4_moe_runtime.protocols.provider import WeightProviderProtocol
from dsv4_moe_runtime.routing.hash_router import HashRouteResolver
from dsv4_moe_runtime.routing.score_router import ScoreRouteResolver

SCORE_TABLE_ROWS = 64


@dataclass
class StepReport:
    """Diagnostics of one decode step (host-side counters only)."""

    layer_hits: int
    layer_misses: int
    layers_executed: int


class DraftInferenceEngine:
    """Single-token draft decoder over the static expert slot pool.

    The attention leg stays on the caller-supplied backend; the MoE leg runs
    through routing -> slot acquisition -> per-projection quantized GEMMs for
    all ``num_layers`` blocks, hash-routed for layers 0-2 and score-gated
    beyond.
    """

    def __init__(
        self,
        config: DeepSeekV4MoEConfig,
        layout: ExpertTensorLayout,
        provider: WeightProviderProtocol,
        backend: DraftBackend,
        num_slots: int,
        device: str,
        attention_heads: int = 8,
    ):
        self._config = config
        self._layout = layout
        self._provider = provider
        self._backend = backend
        self._device = device
        self._attention_heads = attention_heads

        # AOT tables: synthetic tid2eid stack (unique experts per token row --
        # the pool rejects duplicate top-k), gate bias, gate scores, embedding.
        generator = torch.Generator().manual_seed(97)
        layer_offset = torch.arange(config.num_hash_layers, dtype=torch.int64).view(-1, 1, 1) * 5
        token_offset = torch.arange(config.vocab_size, dtype=torch.int64).view(1, -1, 1) * 7
        expert_offset = torch.arange(config.top_k, dtype=torch.int64).view(1, 1, -1) * 3
        self._tid2eid = ((layer_offset + token_offset + expert_offset) % config.num_routed_experts).to(torch.int32)
        self._gate_bias = torch.zeros(config.num_layers, config.num_routed_experts, dtype=torch.float32, device=device)
        self._score_table = torch.randn(SCORE_TABLE_ROWS, config.num_routed_experts, generator=generator)
        self._embedding = torch.empty(config.vocab_size, config.hidden_size, dtype=torch.bfloat16, device=device)
        self._final_norm = torch.ones(config.hidden_size, dtype=torch.float32, device=device)
        self._final_norm_bf16 = self._final_norm.to(torch.bfloat16)

        self._pool = StaticExpertSlotPool(config, num_slots, layout=layout, device=device)
        self._scratchpad = DecodeScratchpad(
            config,
            ScratchpadShapes(tokens=1, num_routed_experts=config.num_routed_experts, top_k=config.top_k),
            device,
        )
        self._runner = MoELayerRunner(
            config=config,
            layout=layout,
            pool=self._pool,
            scratchpad=self._scratchpad,
            backend=backend,
            hash_resolver=HashRouteResolver(config, self._tid2eid),
            score_resolver=ScoreRouteResolver(config, self._gate_bias, max_batch_tokens=1),
        )
        self._fingerprint = self._scratchpad.fingerprint() + [self._pool.slot_arena.data_ptr()]

    # ------------------------------------------------------------------ state

    @property
    def pool(self) -> StaticExpertSlotPool:
        return self._pool

    def arena_fingerprint(self) -> list[int]:
        """data_ptr of scratchpad buffers + pool arena (must never move)."""
        return self._scratchpad.fingerprint() + [self._pool.slot_arena.data_ptr()]

    @property
    def initial_fingerprint(self) -> list[int]:
        return self._fingerprint

    # ------------------------------------------------------------------- step

    def prepare_step(self, token_id: int) -> None:
        """Stage one token: embed it and refresh the synthetic gate scores."""
        if not 0 <= token_id < self._config.vocab_size:
            raise ValueError(f"token_id {token_id} outside [0, {self._config.vocab_size})")
        scratchpad = self._scratchpad
        scratchpad["token_ids"][0] = token_id
        scratchpad["hidden"][0].copy_(self._embedding[token_id])
        scratchpad["route_scores"][0].copy_(self._score_table[token_id % SCORE_TABLE_ROWS])

    def step(self) -> StepReport:
        """Run all layers for the staged token; hidden holds the final state."""
        hits = 0
        misses = 0
        for layer_idx in range(self._config.num_layers):
            layer_hits, layer_misses = self._runner.run_layer(layer_idx, self._provider)
            hits += layer_hits
            misses += layer_misses
            self._attend(layer_idx)
        self._scratchpad["hidden"][0].mul_(self._final_norm_bf16)
        return StepReport(layer_hits=hits, layer_misses=misses, layers_executed=self._config.num_layers)

    def _attend(self, layer_idx: int) -> None:
        """Attention leg + residual, both into pre-allocated buffers."""
        self._backend.fused_attention(self._scratchpad["hidden"], self._scratchpad["attn_out"], layer_idx)
        self._scratchpad["hidden"][0].add_(self._scratchpad["attn_out"][0])

    def warm_up_routing_tables(self) -> None:
        """AOT: resolve one step so lazily-initialized kernel state lands now."""
        self.prepare_step(0)
        self.step()
