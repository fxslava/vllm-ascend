"""Pre-allocated decode scratchpad for the draft inference engine.

Memory contract: every buffer is allocated once here (AOT). A decode step only
writes into these tensors -- ``fingerprint()`` enumerates their data pointers
so the dry run can prove none of them ever moves.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..core.config import DeepSeekV4MoEConfig

ACTIVATION_DTYPE = torch.bfloat16
ACCUMULATOR_DTYPE = torch.float32


@dataclass(frozen=True)
class ScratchpadShapes:
    """Static geometry of one decode step (single token per launch)."""

    tokens: int = 1
    num_routed_experts: int = 256
    top_k: int = 6


class DecodeScratchpad:
    """All per-step activations for the draft decode path.

    Buffer map (T = tokens, K = top_k, D = hidden, I = moe intermediate,
    E = num_routed_experts):

    ================  =================  ==========================
    buffer            shape              purpose
    ================  =================  ==========================
    hidden            [T, D] bf16        running layer input/output
    attn_out          [T, D] bf16        attention projection target
    gathered_x        [K, D] bf16        activations per selected expert
    gate_out          [K, I] bf16        w1 (gate) GEMM target
    up_out            [K, I] bf16        w3 (up) GEMM target
    activated         [K, I] bf16        silu(gate) * up, in place
    down_out          [K, D] bf16        w2 (down) GEMM target
    down_row_f32      [K, D] fp32       dtype bridge for the accumulate
    routed_accum      [T, D] fp32       routed expert sum
    route_experts     [T, K] int32       resolver output
    route_scores      [T, E] fp32        gate scores (score layers)
    route_weights     [K] fp32           normalized routing weights
    token_ids         [T] int64          hash-routing lookup keys
    ================  =================  ==========================
    """

    def __init__(self, config: DeepSeekV4MoEConfig, shapes: ScratchpadShapes, device: str):
        self._tensors: dict[str, torch.Tensor] = {
            "hidden": torch.empty(shapes.tokens, config.hidden_size, dtype=ACTIVATION_DTYPE, device=device),
            "attn_out": torch.empty(shapes.tokens, config.hidden_size, dtype=ACTIVATION_DTYPE, device=device),
            "gathered_x": torch.empty(shapes.top_k, config.hidden_size, dtype=ACTIVATION_DTYPE, device=device),
            "gate_out": torch.empty(shapes.top_k, config.moe_intermediate_size, dtype=ACTIVATION_DTYPE, device=device),
            "up_out": torch.empty(shapes.top_k, config.moe_intermediate_size, dtype=ACTIVATION_DTYPE, device=device),
            "activated": torch.empty(shapes.top_k, config.moe_intermediate_size, dtype=ACTIVATION_DTYPE, device=device),
            "down_out": torch.empty(shapes.top_k, config.hidden_size, dtype=ACTIVATION_DTYPE, device=device),
            "down_row_f32": torch.empty(shapes.top_k, config.hidden_size, dtype=ACCUMULATOR_DTYPE, device=device),
            "routed_accum": torch.empty(shapes.tokens, config.hidden_size, dtype=ACCUMULATOR_DTYPE, device=device),
            "route_experts": torch.empty(shapes.tokens, shapes.top_k, dtype=torch.int32, device=device),
            "route_scores": torch.empty(
                shapes.tokens, shapes.num_routed_experts, dtype=ACCUMULATOR_DTYPE, device=device
            ),
            "route_weights": torch.empty(shapes.top_k, dtype=ACCUMULATOR_DTYPE, device=device),
            "token_ids": torch.empty(shapes.tokens, dtype=torch.int64, device=device),
        }

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._tensors[name]

    def fingerprint(self) -> list[int]:
        return [tensor.data_ptr() for tensor in self._tensors.values()]

    def buffer_names(self) -> list[str]:
        return list(self._tensors)
