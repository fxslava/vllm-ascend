"""Bias-shifted score top-k gating for the DeepSeek-V4 score layers (3+)."""

from __future__ import annotations

import torch

from ..core.config import DeepSeekV4MoEConfig


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
