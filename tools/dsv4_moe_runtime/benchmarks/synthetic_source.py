"""Deterministic synthetic expert bytes for the exclusive staging benchmark.

The stress harness measures the memory subsystem only, so the authoritative
byte source is a seeded pattern generator: every (layer, expert) owns a small
unique tile that ``fill_slot`` broadcasts across the whole slot region and
``read_param`` regenerates for post-run byte verification. Content is distinct
per expert (cross-expert mixups are detectable) and deterministic across runs
(same seed -> same bytes). The tile table is AOT (704 KiB for the full
43x256 model); runtime fills are in-place broadcasts into the transit window
region -- no allocation, no disk.
"""

from __future__ import annotations

import torch

from ..core.layout import ExpertTensorLayout

SYNTHETIC_TILE_BYTES = 64


class SyntheticExpertSource:
    """Seeded per-expert tile broadcast over the slot region (WeightByteSource)."""

    def __init__(self, layout: ExpertTensorLayout, num_layers: int, num_experts: int, seed: int = 0):
        for spec in layout.specs:
            if spec.num_bytes % SYNTHETIC_TILE_BYTES:
                raise ValueError(
                    f"{spec.param_key}: {spec.num_bytes} bytes not divisible by the "
                    f"{SYNTHETIC_TILE_BYTES}-byte synthetic tile"
                )
        self._layout = layout
        self._num_layers = num_layers
        self._num_experts = num_experts
        generator = torch.Generator().manual_seed(seed)
        self._tiles = torch.randint(
            0, 256, (num_layers * num_experts, SYNTHETIC_TILE_BYTES), dtype=torch.uint8, generator=generator
        )

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        return 0 <= layer_idx < self._num_layers and 0 <= expert_id < self._num_experts

    def fill_slot(self, destination: torch.Tensor, layer_idx: int, expert_id: int) -> None:
        if not self.contains(layer_idx, expert_id):
            raise KeyError(f"(layer={layer_idx}, expert={expert_id}) outside synthetic coverage")
        if destination.numel() != self._layout.slot_num_bytes:
            raise ValueError(
                f"destination holds {destination.numel()} bytes, slot layout is {self._layout.slot_num_bytes}"
            )
        destination.view(-1, SYNTHETIC_TILE_BYTES).copy_(self._tile(layer_idx, expert_id))

    def read_param(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor:
        if not self.contains(layer_idx, expert_id):
            raise KeyError(f"(layer={layer_idx}, expert={expert_id}) outside synthetic coverage")
        spec = self._layout.spec_for(param_key)
        repeats = spec.num_bytes // SYNTHETIC_TILE_BYTES
        return self._tile(layer_idx, expert_id).repeat(repeats).view(spec.view_shape)

    def _tile(self, layer_idx: int, expert_id: int) -> torch.Tensor:
        return self._tiles[layer_idx * self._num_experts + expert_id]
