"""Deterministic ``tid2eid`` routing for the DeepSeek-V4 hash layers (0-2)."""

from __future__ import annotations

import torch

from ..core.config import DeepSeekV4MoEConfig


class HashRouteResolver:
    """Deterministic ``tid2eid`` routing for layers ``[0, num_hash_layers)``.

    Matches ``Gate.forward`` in the reference implementation: expert indices
    are a pure function of the token id -- ``indices = tid2eid[input_ids]`` --
    and each hash layer owns its own table (the checkpoint ships one
    ``layers.<i>.ffn.gate.tid2eid`` per hash layer). Accepts either a shared
    ``[vocab, top_k]`` table or the full per-layer ``[num_hash_layers, vocab,
    top_k]`` stack. Allocation-free: ``torch.index_select(..., out=)`` writes
    caller storage. ``out`` must be ``int32 [num_tokens, top_k]``;
    ``network_input`` is the ``int64`` token-id vector.
    """

    def __init__(self, config: DeepSeekV4MoEConfig, tid2eid: torch.Tensor):
        shared_shape = (config.vocab_size, config.top_k)
        layered_shape = (config.num_hash_layers, config.vocab_size, config.top_k)
        if tuple(tid2eid.shape) not in (shared_shape, layered_shape):
            raise ValueError(
                f"tid2eid must be {shared_shape} (shared) or {layered_shape} (per-layer), got {tuple(tid2eid.shape)}"
            )
        if tid2eid.dtype != torch.int32:
            raise ValueError(f"tid2eid must be int32 to mirror the checkpoint, got {tid2eid.dtype}")
        if int(tid2eid.max()) >= config.num_routed_experts:
            raise ValueError("tid2eid references an expert outside [0, num_routed_experts)")
        self._config = config
        self._tid2eid = tid2eid

    def resolve(self, layer_idx: int, network_input: torch.Tensor, out: torch.Tensor) -> None:
        if layer_idx >= self._config.num_hash_layers:
            raise ValueError(f"layer {layer_idx} is score-gated (hash layers are [0, {self._config.num_hash_layers}))")
        if network_input.dtype != torch.int64:
            raise ValueError(f"token ids must be int64, got {network_input.dtype}")
        table = self._tid2eid if self._tid2eid.ndim == 2 else self._tid2eid[layer_idx]
        torch.index_select(table, 0, network_input, out=out)
