"""Route resolver dependency boundary: per-layer token/expert to top-k ids."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class RouteResolverProtocol(Protocol):
    """Per-layer routing policy: fills ``out`` in place (hash vs score gating).

    Implementations own one layer family; ``network_input`` semantics differ
    (token ids for hash layers, gate scores for score layers) and each
    implementation documents its expected ``out`` dtype.
    """

    def resolve(self, layer_idx: int, network_input: torch.Tensor, out: torch.Tensor) -> None: ...
