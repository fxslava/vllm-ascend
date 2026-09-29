"""Standalone V5 operator wrappers with pre-allocated workspace management.

Two operators back the DeepSeek-V4 draft decode path (Ascend 950PR, aclnn V5
exclusively -- EZ9903 avoidance):

* :class:`FusedInferAttentionScoreV5` -- paged MLA decode attention. The
  46-argument plan signature is transcribed verbatim from
  ``csrc/tests/common/aclnn_ops_950pr.hpp``; the launch entry is resolved from
  the V5 candidates (never from the withdrawn V1-V4 family).
* :class:`WeightQuantBatchMatmulOp` -- per-block quantized GEMM for the expert
  projections (w1/w2/w3): FP4-packed weight bytes, E8M0 per-block-32 scales,
  dequantized on the fly against bf16 activations.

Workspace mechanics (memory contract): both operators follow CANN's two-call
protocol. :meth:`plan_static` runs once at init with the real static shapes and
allocates the workspace buffer **once**; :meth:`execute` re-plans per launch
(a host-only call -- CANN executors are single-shot) and *refuses* to grow the
workspace, so a decode step allocates nothing on the device.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

import torch

from dsv4_moe_runtime.hardware.aclnn_binding import (
    AclIntArrayHandle,
    AclnnLibrary,
    AclTensorHandle,
)

_LAUNCH_STATUS_OK = 0

# Plan argument kinds for the FIA V5 GetWorkspaceSize entry, transcribed from
# csrc/tests/common/aclnn_ops_950pr.hpp (31 tensor/int-array pointers, then 15
# scalars, then the two out-tensors, then workspace size + executor).
_FIA_V5_PLAN_ARGTYPES: list[object] = (
    [ctypes.c_void_p] * 31
    + [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_bool,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    + [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
)

# Weight-quant batch matmul plan kinds (CANN aclnn_weight_quant_batch_matmul
# V2 family): x, weight, antiquant_scale, antiquant_offset, bias, quant_scale,
# quant_offset, transpose_weight, dst_type, out, workspace_size, executor.
# The inventory probe reports which candidate the deployed CANN exposes;
# extend the list from its header at bring-up if it ships a newer entry.
_WEIGHT_QUANT_PLAN_ARGTYPES: list[object] = (
    [ctypes.c_void_p] * 7
    + [ctypes.c_bool, ctypes.c_int64, ctypes.c_void_p]
    + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
)
_WEIGHT_QUANT_PLAN_CANDIDATES = (
    "aclnnWeightQuantBatchMatmulV2GetWorkspaceSize",
    "aclnnWeightQuantBatchMatmulGetWorkspaceSize",
)


@dataclass(frozen=True)
class FiaV5DecodeConfig:
    """Scalar configuration block for one paged MLA decode attention call."""

    num_heads: int
    num_key_value_heads: int = 1
    scale_value: float = 0.0  # 0 -> the kernel uses 1/sqrt(head_dim)
    pre_tokens: int = 65536
    next_tokens: int = 0  # causal decode convention (see tq_longbench.ops)
    input_layout: str = "TND"
    sparse_mode: int = 3
    inner_precise: int = 0
    block_size: int = 128
    antiquant_mode: int = 0
    softmax_lse_flag: bool = False
    key_antiquant_mode: int = 0
    value_antiquant_mode: int = 0
    query_quant_mode: int = 0
    pse_type: int = 0


@dataclass(frozen=True)
class WeightQuantGemmConfig:
    transpose_weight: bool = True  # weights stored [out, k], matching the slot views
    dst_dtype_code: int = 27  # ACL_DT_BF16


@dataclass
class _PlanResult:
    workspace_size: int
    executor: ctypes.c_void_p


class _V5OpBase:
    """Two-call mechanics: AOT workspace reservation + single-shot executors."""

    plan_symbol: str = ""
    launch_candidates: tuple[str, ...] = ()
    plan_argtypes: list[object] = []

    def __init__(self, library: AclnnLibrary, device: str, workspace_bytes: int = 0):
        if not library.loaded:
            raise RuntimeError(
                "libopapi.so is not loaded; V5 operators require the CANN runtime "
                "(bring up on the Ascend host, or use the CPU mock backend)"
            )
        self._library = library
        self._device = device
        plan_fn = library.resolve(self.plan_symbol)
        if plan_fn is None:
            raise RuntimeError(f"{self.plan_symbol} not found in the loaded opapi library")
        prototype = ctypes.CFUNCTYPE(ctypes.c_int, *self.plan_argtypes)
        self._plan_fn = prototype(plan_fn)
        self._launch_fn = self._resolve_launch()
        # AOT workspace: allocated once here, never during decode.
        self._workspace = torch.empty(max(workspace_bytes, 1), dtype=torch.uint8, device=device)
        self._workspace_capacity = self._workspace.numel()

    def _resolve_launch(self):
        for candidate in self.launch_candidates:
            launch = self._library.resolve(candidate)
            if launch is not None:
                launch.restype = ctypes.c_int
                launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
                return launch
        raise RuntimeError(f"no launch entry found for {self.plan_symbol}; tried {self.launch_candidates}")

    def _plan(self, *args: object) -> _PlanResult:
        workspace_size = ctypes.c_uint64(0)
        executor = ctypes.c_void_p(0)
        status = self._plan_fn(*args, ctypes.byref(workspace_size), ctypes.byref(executor))
        if status != 0:
            raise RuntimeError(f"{self.plan_symbol} failed with aclnnStatus {status}")
        return _PlanResult(workspace_size.value, executor)

    def _reserve_workspace(self, num_bytes: int) -> None:
        """Init-time reservation (the only allocation this wrapper ever makes)."""
        if num_bytes > self._workspace_capacity:
            self._workspace = torch.empty(max(num_bytes, 1), dtype=torch.uint8, device=self._device)
            self._workspace_capacity = self._workspace.numel()

    def _assert_reserved(self, num_bytes: int) -> None:
        if num_bytes > self._workspace_capacity:
            raise RuntimeError(
                f"plan returned {num_bytes} workspace bytes but only {self._workspace_capacity} "
                "were reserved at init; decode shapes changed -- re-create the operator (AOT), "
                "never grow the workspace inside a decode step"
            )

    def _launch(self, workspace_size: int, executor: ctypes.c_void_p, stream_pointer: int | None) -> None:
        workspace_pointer = ctypes.c_void_p(self._workspace.data_ptr()) if workspace_size > 0 else ctypes.c_void_p(0)
        status = self._launch_fn(workspace_pointer, workspace_size, executor, stream_pointer)
        if status != _LAUNCH_STATUS_OK:
            raise RuntimeError(f"{type(self).__name__} launch failed with aclnnStatus {status}")


class FusedInferAttentionScoreV5(_V5OpBase):
    """Paged decode attention through ``aclnnFusedInferAttentionScoreV5``."""

    plan_symbol = "aclnnFusedInferAttentionScoreV5GetWorkspaceSize"
    launch_candidates = ("aclnnFusedInferAttentionScoreV5", "aclnnFusedInferAttentionScore")
    plan_argtypes = _FIA_V5_PLAN_ARGTYPES

    def __init__(
        self,
        library: AclnnLibrary,
        config: FiaV5DecodeConfig,
        device: str,
        workspace_bytes: int = 0,
    ):
        super().__init__(library, device, workspace_bytes)
        self._config = config
        self._layout_buffer = ctypes.create_string_buffer(config.input_layout.encode("ascii"))

    @staticmethod
    def _null() -> ctypes.c_void_p:
        return ctypes.c_void_p(0)

    def _plan_args(
        self,
        query: AclTensorHandle,
        key_list_pointer: ctypes.c_void_p,
        value_list_pointer: ctypes.c_void_p,
        attention_out: AclTensorHandle,
        softmax_lse: AclTensorHandle,
        actual_seq_q: AclIntArrayHandle,
        actual_seq_kv: AclIntArrayHandle,
        block_table: AclTensorHandle | None,
    ) -> tuple[object, ...]:
        cfg = self._config
        null = self._null()
        return (
            query.pointer,
            key_list_pointer,
            value_list_pointer,
            null,
            null,
            actual_seq_q.pointer,
            actual_seq_kv.pointer,
            null,
            null,
            null,
            null,
            null,  # deq/quant scale+offset block
            null,
            null,  # antiquant scale/offset
            block_table.pointer if block_table else null,
            null,
            null,  # query/kv padding size
            null,
            null,
            null,
            null,  # key/value antiquant scale+offset
            null,
            null,
            null,  # shared prefix k/v + len
            null,
            null,
            null,  # query_rope, key_rope, key_rope_antiquant_scale
            null,
            null,  # dequant_scale_query, learnable_sink
            null,
            null,  # q_start_idx, kv_start_idx
            cfg.num_heads,
            cfg.scale_value,
            cfg.pre_tokens,
            cfg.next_tokens,
            self._layout_buffer,
            cfg.num_key_value_heads,
            cfg.sparse_mode,
            cfg.inner_precise,
            cfg.block_size,
            cfg.antiquant_mode,
            cfg.softmax_lse_flag,
            cfg.key_antiquant_mode,
            cfg.value_antiquant_mode,
            cfg.query_quant_mode,
            cfg.pse_type,
            attention_out.pointer,
            softmax_lse.pointer,
        )

    def plan_static(
        self,
        query: AclTensorHandle,
        key_list_pointer: ctypes.c_void_p,
        value_list_pointer: ctypes.c_void_p,
        attention_out: AclTensorHandle,
        softmax_lse: AclTensorHandle,
        actual_seq_q: AclIntArrayHandle,
        actual_seq_kv: AclIntArrayHandle,
        block_table: AclTensorHandle | None,
    ) -> _PlanResult:
        """Init-time plan with the real static shapes; reserves the workspace."""
        result = self._plan(
            *self._plan_args(
                query,
                key_list_pointer,
                value_list_pointer,
                attention_out,
                softmax_lse,
                actual_seq_q,
                actual_seq_kv,
                block_table,
            )
        )
        self._reserve_workspace(result.workspace_size)
        return result

    def execute(
        self,
        query: AclTensorHandle,
        key_list_pointer: ctypes.c_void_p,
        value_list_pointer: ctypes.c_void_p,
        attention_out: AclTensorHandle,
        softmax_lse: AclTensorHandle,
        actual_seq_q: AclIntArrayHandle,
        actual_seq_kv: AclIntArrayHandle,
        block_table: AclTensorHandle | None,
        stream_pointer: int | None,
    ) -> None:
        """Decode-step launch: host-only re-plan into the reserved workspace."""
        result = self._plan(
            *self._plan_args(
                query,
                key_list_pointer,
                value_list_pointer,
                attention_out,
                softmax_lse,
                actual_seq_q,
                actual_seq_kv,
                block_table,
            )
        )
        self._assert_reserved(result.workspace_size)
        self._launch(result.workspace_size, result.executor, stream_pointer)


class WeightQuantBatchMatmulOp(_V5OpBase):
    """Per-block quantized GEMM for expert projections (FP4 weights + E8M0 scales)."""

    plan_symbol = "aclnnWeightQuantBatchMatmulV2GetWorkspaceSize"
    launch_candidates = ("aclnnWeightQuantBatchMatmulV2", "aclnnWeightQuantBatchMatmul")
    plan_argtypes = _WEIGHT_QUANT_PLAN_ARGTYPES

    def __init__(
        self,
        library: AclnnLibrary,
        config: WeightQuantGemmConfig,
        device: str,
        workspace_bytes: int = 0,
    ):
        resolved = next((s for s in _WEIGHT_QUANT_PLAN_CANDIDATES if library.resolve(s) is not None), None)
        if resolved is None:
            raise RuntimeError(
                "no weight-quant batch matmul plan symbol found; tried "
                f"{_WEIGHT_QUANT_PLAN_CANDIDATES}. The deployed CANN on this 950PR host "
                "may expose a newer V5-family entry -- extend the candidate list after "
                "running AclnnLibrary.inventory()."
            )
        self.plan_symbol = resolved
        super().__init__(library, device, workspace_bytes)
        self._config = config

    def plan_static(
        self,
        x: AclTensorHandle,
        weight: AclTensorHandle,
        antiquant_scale: AclTensorHandle,
        out: AclTensorHandle,
    ) -> _PlanResult:
        """Init-time plan; antiquant offsets/bias/quant scales stay null."""
        result = self._plan(
            x.pointer,
            weight.pointer,
            antiquant_scale.pointer,
            self._null(),
            self._null(),
            self._null(),
            self._null(),
            self._config.transpose_weight,
            self._config.dst_dtype_code,
            out.pointer,
        )
        self._reserve_workspace(result.workspace_size)
        return result

    def execute(
        self,
        x: AclTensorHandle,
        weight: AclTensorHandle,
        antiquant_scale: AclTensorHandle,
        out: AclTensorHandle,
        stream_pointer: int | None,
    ) -> None:
        result = self._plan(
            x.pointer,
            weight.pointer,
            antiquant_scale.pointer,
            self._null(),
            self._null(),
            self._null(),
            self._null(),
            self._config.transpose_weight,
            self._config.dst_dtype_code,
            out.pointer,
        )
        self._assert_reserved(result.workspace_size)
        self._launch(result.workspace_size, result.executor, stream_pointer)

    @staticmethod
    def _null() -> ctypes.c_void_p:
        return ctypes.c_void_p(0)


def resolve_stream_pointer(stream: object | None) -> int | None:
    """Raw ``aclrtStream`` from a torch_npu stream object (bring-up contract)."""
    if stream is None:
        return None
    pointer = getattr(stream, "npu_stream", None)
    return int(pointer) if pointer is not None else None
