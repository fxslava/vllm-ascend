"""Weight provider dependency boundaries (DIP).

The slot pool never knows where expert weights come from (safetensors mmap,
KV-to-DDR spiller, loopback test buffer); it interacts only with these
interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch

from .residency_policy import AdmissionDecision


@runtime_checkable
class WeightProviderProtocol(Protocol):
    """Blocking provider interface.

    The pool asks for the pinned host tensor of one expert parameter and
    performs the single in-place ``slot_view.copy_(pinned, non_blocking=False)``
    itself. Implementations must return pre-allocated, contiguous, CPU tensors
    of the exact shape ``ExpertTensorLayout.spec_for(param_key).view_shape`` --
    serving may not allocate.
    """

    def pinned_cpu_weight(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor: ...


@runtime_checkable
class SlotFillProviderProtocol(Protocol):
    """Extended provider interface for streamed DMA transports.

    Host-pinned-DDR-to-NPU-HBM staging (``copy_(..., non_blocking=True)`` on a
    dedicated copy stream) owns its stream and sync semantics, so such providers
    write the pool's pre-sliced slot views themselves and report the bytes
    moved. Refines :class:`WeightProviderProtocol` -- implementations must
    satisfy both; declared standalone (not protocol-inherited) so it stays
    runtime-checkable across Python versions. Contracts:

    * ``ensure_staged`` must raise (e.g. ``KeyError``) *before* any byte moves,
      which lets the pool keep acquisition transactional;
    * ``fill_slot_params`` fills the given views strictly in place (never
      resizes or replaces them) and returns the number of bytes staged.
    """

    def pinned_cpu_weight(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor: ...

    def ensure_staged(self, layer_idx: int, expert_id: int) -> None: ...

    def fill_slot_params(self, layer_idx: int, expert_id: int, views: Mapping[str, torch.Tensor]) -> int: ...


@runtime_checkable
class WeightByteSource(Protocol):
    """Authoritative expert byte source for exclusive staging windows.

    A bounded transit window streams experts through on demand: on a window
    miss it reserves a pinned slot and asks the source to materialize the
    expert's bytes into it (``fill_slot``), and post-run byte verification
    re-materializes single parameters (``read_param``). The source is
    authoritative by definition -- window entries are droppable caches of it,
    never the only copy -- which is what makes bounded ring eviction safe.
    """

    def contains(self, layer_idx: int, expert_id: int) -> bool:
        """Whether the source can serve this expert (cheap, mutation-free)."""
        ...

    def fill_slot(self, destination: torch.Tensor, layer_idx: int, expert_id: int) -> None:
        """Write the full ``slot_num_bytes`` expert slot into the pinned region."""
        ...

    def read_param(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor:
        """Materialize one parameter's expected bytes (verification path)."""
        ...


@runtime_checkable
class ExclusiveSwapProviderProtocol(Protocol):
    """Own the RAM partition and exchange the policy-selected VRAM victims."""

    def exchange_admissions(self, decision: AdmissionDecision, pool: object) -> int: ...

    def validate_residency(self, pool: object) -> None: ...
