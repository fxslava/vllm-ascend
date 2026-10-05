"""Draft-path compute backends: V5 ctypes wrapper (NPU) and CPU contract mock.

The backend seam keeps the draft engine device-agnostic (DIP): the engine
issues ``quant_gemm`` / ``fused_attention`` calls against pre-allocated
buffers; the backend either drives the real aclnn V5 entry points via ctypes
(:class:`AclnnV5Backend`, Ascend 950PR only) or validates the same shape
contracts without arithmetic (:class:`MockV5Backend`, CPU dry run).

Both backends are write-only into caller storage: no buffer they touch is
ever allocated at call time.
"""

from __future__ import annotations

from typing import Protocol

import torch

from ..hardware.aclnn_binding import AclnnLibrary
from ..hardware.runtime import DeviceRuntime
from ..hardware.v5_ops import (
    FiaV5DecodeConfig,
    FusedInferAttentionScoreV5,
    WeightQuantBatchMatmulOp,
    WeightQuantGemmConfig,
    resolve_stream_pointer,
)


class DraftBackend(Protocol):
    """Shape-typed compute seam consumed by the draft layer runner."""

    def quant_gemm(
        self, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, out: torch.Tensor, label: str
    ) -> None: ...

    def fused_attention(self, hidden: torch.Tensor, out: torch.Tensor, layer_idx: int) -> None: ...


class MockV5Backend:
    """CPU contract mock: validates shapes/dtypes, writes deterministic bytes.

    Honest scope: this backend performs **no arithmetic fidelity** -- it proves
    plumbing (operand geometry, dtype flow, write-only targets) so that the
    exact numerics can be golden-tested on the 950PR against the real V5
    kernels, like ``csrc/tests`` does for the TurboQuant stages.
    """

    def quant_gemm(
        self, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, out: torch.Tensor, label: str
    ) -> None:
        out_rows, out_cols = out.shape
        if x.shape[0] != out_rows:
            raise ValueError(f"{label}: x rows {x.shape[0]} != out rows {out_rows}")
        if x.shape[1] // 2 != weight.shape[1]:
            raise ValueError(f"{label}: packed weight K dim {weight.shape[1]} does not match x K {x.shape[1]}")
        if weight.shape[0] != out_cols:
            raise ValueError(f"{label}: weight rows {weight.shape[0]} != out cols {out_cols}")
        if scale.shape[0] != weight.shape[0] or scale.shape[1] != x.shape[1] // 32:
            raise ValueError(f"{label}: scale {tuple(scale.shape)} does not match block-32 of the weight")
        pattern = (out_cols % 251 + 1) / 257.0
        out.fill_(pattern)

    def fused_attention(self, hidden: torch.Tensor, out: torch.Tensor, layer_idx: int) -> None:
        if hidden.shape != out.shape:
            raise ValueError(f"layer {layer_idx}: attention out {tuple(out.shape)} != hidden {tuple(hidden.shape)}")
        out.fill_((layer_idx % 31 + 1) / 33.0)


class AclnnV5Backend:
    """Ascend 950PR backend: ctypes calls into aclnnFusedInferAttentionScoreV5
    and the weight-quant batch matmul family.

    Bring-up contract: requires ``libopapi.so`` on the host and an initialized
    torch_npu device; the inventory is asserted V5-only at construction
    (EZ9903). Descriptors are (re)built per call from the scratchpad/slot-view
    tensors -- host-side objects only; device memory stays untouched.
    """

    def __init__(self, runtime: DeviceRuntime, num_heads: int, workspace_bytes: int = 4 * 1024 * 1024):
        self._runtime = runtime
        self._library = AclnnLibrary()
        inventory = self._library.inventory()
        if not inventory.library_loaded:
            raise RuntimeError(f"cannot bring up the V5 backend: {inventory.load_error}")
        inventory.assert_v5_only()
        self._gemm = WeightQuantBatchMatmulOp(
            self._library, WeightQuantGemmConfig(), device=runtime.device, workspace_bytes=workspace_bytes
        )
        self._attention = FusedInferAttentionScoreV5(
            self._library,
            FiaV5DecodeConfig(num_heads=num_heads),
            device=runtime.device,
            workspace_bytes=workspace_bytes,
        )

    def quant_gemm(
        self, x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, out: torch.Tensor, label: str
    ) -> None:
        x_handle = self._library.create_tensor(x)
        weight_handle = self._library.create_tensor(weight)
        scale_handle = self._library.create_tensor(scale)
        out_handle = self._library.create_tensor(out)
        try:
            self._gemm.execute(x_handle, weight_handle, scale_handle, out_handle, stream_pointer=self._stream())
        finally:
            self._library.destroy_tensor(x_handle)
            self._library.destroy_tensor(weight_handle)
            self._library.destroy_tensor(scale_handle)
            self._library.destroy_tensor(out_handle)

    def fused_attention(self, hidden: torch.Tensor, out: torch.Tensor, layer_idx: int) -> None:
        """Paged MLA decode attention (draft: contiguous KV list, no pagers).

        Bring-up boundary: the full V5 call needs live K/V tensor lists, the
        TND sequence-length arrays and a device to validate descriptor layouts
        against -- the plan signature is already bound in
        :class:`FusedInferAttentionScoreV5`; finish the descriptor wiring on
        the 950PR host (mirroring ``csrc/tests`` stage 5) before enabling.
        Until then the dry run drives attention through the CPU mock backend.
        """
        raise RuntimeError(
            "AclnnV5Backend.fused_attention is a device bring-up item: wire the "
            "FusedInferAttentionScoreV5 descriptors (K/V tensor lists, TND seq-lens) "
            "on the Ascend host; run the dry run with --attention-backend mock meanwhile"
        )

    def _stream(self) -> int | None:
        # Launch on the same stream as the tensor producers and consumers.
        # A fresh stream per launch races both and discards its lifetime owner.
        return resolve_stream_pointer(self._runtime.current_stream())
