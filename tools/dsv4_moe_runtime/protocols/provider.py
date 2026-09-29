"""Weight provider dependency boundaries (DIP).

The slot pool never knows where expert weights come from (safetensors mmap,
KV-to-DDR spiller, loopback test buffer); it interacts only with these
interfaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import torch


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
