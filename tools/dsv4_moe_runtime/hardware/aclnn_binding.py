"""Standalone ctypes bindings for CANN aclnn V5 entry points (Ascend 950PR).

This module mirrors the C++ harness in ``csrc/tests/common/aclnn_runtime.{hpp,cpp}``
one-to-one, but drives it from Python via ``ctypes`` so the runtime has **zero
dependency on vllm-ascend** and no compiled extension:

* ``libopapi.so`` is loaded from the standard CANN locations (``ASCEND_HOME_PATH``
  / ``ASCEND_TOOLKIT_HOME`` lib64, stubs, or an explicit path override); custom
  op packages are searched first, exactly like ``OpApiLibrary::Resolve``.
* ``aclTensor`` descriptors are built with the C entry points the harness uses:
  ``aclCreateTensor(view_dims, n, dtype, stride, offset, format, storage_dims, n, data)``,
  plus ``aclCreateIntArray`` / ``aclCreateScalar`` / ``aclCreateTensorList``.
* Every operator follows CANN's two-call protocol: the ``<op>GetWorkspaceSize``
  plan entry fills ``(workspace_size, executor)``, then the ``<op>`` launch
  entry runs on an ``aclrtStream`` with a **pre-allocated** workspace buffer.

EZ9903 policy: this binding never resolves or invokes any legacy
FusedInferAttentionScore V1-V4 plan/launch symbol. The inventory only *reports*
whether the withdrawn V2 plan symbol is (incorrectly) present on the deployed
stack, so bring-up can assert the 950PR contract.

Signatures were extracted from ``csrc/tests/common/aclnn_ops_950pr.hpp`` (V5
attention) and the CANN ``aclnn_weight_quant_batch_matmul`` family (per-block
FP4/E8M0 expert GEMM). The dtype/format enum values below mirror
``acl_base.h``; :meth:`AclnnLibrary.inventory` verifies the deployed CANN at
bring-up.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

try:
    import torch
except ImportError:  # pure-ctypes bring-up (sim containers ship no torch)
    torch = None

LIBOPAPI_NAME = "libopapi.so"
LIBASCENDCL_NAME = "libascendcl.so"

# aclFormat (acl_base.h): ACL_FORMAT_ND
ACL_FORMAT_ND = 2

# aclDataType (CANN acl_base_rt.h, verified against 9.2.0-beta.2): codes used by
# the V5 decode/MoE path. FP8 codes 35/36 and the UE8M0 block-scale code 37 are
# fixed by the toolkit ABI (GroupedDynamicMxQuant doc: {35: FLOAT8_E5M2,
# 36: FLOAT8_E4M3FN}); 40/41 are the packed FP4 pairs.
ACL_DT_FLOAT32 = 0
ACL_DT_FLOAT16 = 1
ACL_DT_INT8 = 2
ACL_DT_INT32 = 3
ACL_DT_UINT8 = 4
ACL_DT_INT64 = 9
ACL_DT_BF16 = 27
ACL_DT_FLOAT8_E5M2 = 35
ACL_DT_FLOAT8_E4M3FN = 36
ACL_DT_FLOAT8_E8M0 = 37
ACL_DT_FP4X2_E2M1 = 40
ACL_DT_FP4X2_E1M2 = 41

# Built lazily: keys live on torch.dtype, and bring-up hosts may have no torch.
_TORCH_TO_ACL_DTYPE: dict[object, int] = {}


def _build_torch_dtype_map() -> None:
    if _TORCH_TO_ACL_DTYPE or torch is None:
        return
    mapping = {
        torch.float32: ACL_DT_FLOAT32,
        torch.float16: ACL_DT_FLOAT16,
        torch.bfloat16: ACL_DT_BF16,
        torch.int8: ACL_DT_INT8,
        torch.int32: ACL_DT_INT32,
        torch.uint8: ACL_DT_UINT8,
        torch.int64: ACL_DT_INT64,
    }
    for attr, code in (
        ("float8_e4m3fn", ACL_DT_FLOAT8_E4M3FN),
        ("float8_e5m2", ACL_DT_FLOAT8_E5M2),
        ("float8_e8m0fnu", ACL_DT_FLOAT8_E8M0),
        ("float4_e2m1fn_x2", ACL_DT_FP4X2_E2M1),
    ):
        dtype = getattr(torch, attr, None)
        if dtype is not None:
            mapping[dtype] = code
    _TORCH_TO_ACL_DTYPE.update(mapping)

_ACL_STATUS_OK = 0


def _contiguous_strides(dims: Sequence[int]) -> tuple[int, ...]:
    """Row-major element strides for ``dims`` (a 0-dim tensor has none)."""
    if not dims:
        return ()
    strides = [0] * len(dims)
    running = 1
    for axis in range(len(dims) - 1, -1, -1):
        strides[axis] = running
        running *= dims[axis]
    return tuple(strides)

# The FIA family on Ascend 950: only the V5 plan/launch symbols may exist and
# may ever be bound (EZ9903 avoidance). Probing records -- never wraps -- them.
FIA_V5_PLAN_SYMBOL = "aclnnFusedInferAttentionScoreV5GetWorkspaceSize"
FIA_V5_LAUNCH_CANDIDATES = ("aclnnFusedInferAttentionScoreV5", "aclnnFusedInferAttentionScore")
LEGACY_FIA_PROBE_SYMBOLS = (
    "aclnnFusedInferAttentionScoreV2GetWorkspaceSize",
    "aclnnFusedInferAttentionScoreGetWorkspaceSize",
)

# Weight-only per-block quantized GEMM family used for the FP4-block32 expert
# projections (x dequantized on the fly with E8M0 scales, weight FP4 packed).
QUANT_MATMUL_PLAN_CANDIDATES = (
    "aclnnWeightQuantBatchMatmulV2GetWorkspaceSize",
    "aclnnWeightQuantBatchMatmulGetWorkspaceSize",
)

# DeepSeek-V4 Flash MoE stack (V5 generation, 950PR): gating -> dispatch ->
# FP4 grouped expert GEMM. Probed with the attention/quant entries at bring-up.
MOE_V5_PLAN_SYMBOLS = (
    "aclnnMoeGatingTopKV2GetWorkspaceSize",
    "aclnnMoeInitRoutingV4GetWorkspaceSize",
    "aclnnGroupedMatmulV5GetWorkspaceSize",
)
MOE_V5_LAUNCH_SYMBOLS = (
    "aclnnMoeGatingTopKV2",
    "aclnnMoeInitRoutingV4",
    "aclnnGroupedMatmulV5",
)


def opapi_candidate_paths() -> list[str]:
    """Load candidates mirroring ``OpApiCandidatePaths`` in the C++ harness."""
    override = os.environ.get("DSV4_LIBOPAPI_PATH", "")
    candidates = [override] if override else []
    candidates.append(LIBOPAPI_NAME)
    ascend_home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME") or ""
    if ascend_home:
        home = Path(ascend_home)
        candidates += [
            str(home / "lib64" / LIBOPAPI_NAME),
            str(home / "lib64" / "stub" / LIBOPAPI_NAME),
            str(home / "aarch64-linux" / "lib64" / LIBOPAPI_NAME),
            str(home / "x86_64-linux" / "lib64" / LIBOPAPI_NAME),
        ]
    return candidates


@dataclass
class AclTensorHandle:
    """Keeps a C ``aclTensor*`` alive together with the arrays it borrows."""

    pointer: ctypes.c_void_p
    view_dims: ctypes.Array
    strides: ctypes.Array
    storage_dims: ctypes.Array


@dataclass
class AclIntArrayHandle:
    """Keeps a C ``aclIntArray*`` alive together with its backing int64 array."""

    pointer: ctypes.c_void_p
    values: ctypes.Array


@dataclass
class AclTensorListHandle:
    """Keeps a C ``aclTensorList*`` alive together with the descriptors it wraps."""

    pointer: ctypes.c_void_p
    pointers: ctypes.Array
    handles: list[AclTensorHandle]


@dataclass
class InventoryEntry:
    symbol: str
    found: bool
    source: str = ""


@dataclass
class AclnnInventory:
    """Bring-up report over the deployed CANN operator surface."""

    library_loaded: bool
    library_path: str = ""
    load_error: str = ""
    entries: list[InventoryEntry] = field(default_factory=list)
    legacy_fia_present: list[str] = field(default_factory=list)

    def assert_v5_only(self) -> None:
        """EZ9903 guard: no legacy FIA plan symbol may be resolvable on 950PR."""
        if self.legacy_fia_present:
            raise RuntimeError(
                "legacy FusedInferAttentionScore plan symbols are resolvable on this stack "
                f"({', '.join(self.legacy_fia_present)}); Ascend 950PR requires the V5 "
                "interface exclusively and must not carry V1-V4"
            )


class AclnnLibrary:
    """ctypes facade over libopapi.so / libascendcl.so (V5 entry points only)."""

    def __init__(self, library_path: str | None = None, ascendcl_path: str | None = None):
        self._opapi = self._load(library_path or opapi_candidate_paths())
        self._ascendcl = self._load([ascendcl_path] if ascendcl_path else [LIBASCENDCL_NAME, ""])
        self.library_path = library_path if (library_path and self._opapi is not None) else ""
        self._custom_handles: list[ctypes.CDLL] = []
        self._configure_signatures()

    # ------------------------------------------------------------- loading

    @staticmethod
    def _load(candidates: list[str]) -> ctypes.CDLL | None:
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
        return None

    @property
    def loaded(self) -> bool:
        return self._opapi is not None

    def load_custom_package(self, path: str | os.PathLike[str]) -> None:
        """Register a custom op package searched before CANN (like the harness)."""
        if self._opapi is None:
            raise RuntimeError("libopapi.so is not loaded; cannot register a custom op package")
        handle = ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        self._custom_handles.append(handle)

    def _configure_signatures(self) -> None:
        if self._opapi is None:
            return
        self._opapi.aclCreateTensor.restype = ctypes.c_void_p
        self._opapi.aclCreateTensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_int64,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
            ctypes.c_void_p,
        ]
        self._opapi.aclDestroyTensor.restype = ctypes.c_int
        self._opapi.aclDestroyTensor.argtypes = [ctypes.c_void_p]
        self._opapi.aclCreateIntArray.restype = ctypes.c_void_p
        self._opapi.aclCreateIntArray.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64]
        self._opapi.aclDestroyIntArray.restype = ctypes.c_int
        self._opapi.aclDestroyIntArray.argtypes = [ctypes.c_void_p]
        self._opapi.aclCreateScalar.restype = ctypes.c_void_p
        self._opapi.aclCreateScalar.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._opapi.aclDestroyScalar.restype = ctypes.c_int
        self._opapi.aclDestroyScalar.argtypes = [ctypes.c_void_p]
        self._opapi.aclCreateTensorList.restype = ctypes.c_void_p
        self._opapi.aclCreateTensorList.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint64]
        self._opapi.aclDestroyTensorList.restype = ctypes.c_int
        self._opapi.aclDestroyTensorList.argtypes = [ctypes.c_void_p]

    # ---------------------------------------------------------- resolution

    def resolve(self, symbol: str) -> ctypes.CDLL._FuncPtr | None:
        if self._opapi is None:
            return None
        for handle in self._custom_handles:
            fn = getattr(handle, symbol, None)
            if fn is not None:
                return fn
        return getattr(self._opapi, symbol, None)

    def inventory(self) -> AclnnInventory:
        """Probe the operator surface; asserts the 950PR V5-only contract."""
        report = AclnnInventory(library_loaded=self.loaded, library_path=self.library_path)
        if not self.loaded:
            report.load_error = "libopapi.so could not be loaded on this host"
            return report
        for symbol in (FIA_V5_PLAN_SYMBOL, *QUANT_MATMUL_PLAN_CANDIDATES, *MOE_V5_PLAN_SYMBOLS):
            found = self.resolve(symbol) is not None
            report.entries.append(InventoryEntry(symbol=symbol, found=found))
        for launch in (*FIA_V5_LAUNCH_CANDIDATES, *MOE_V5_LAUNCH_SYMBOLS):
            report.entries.append(InventoryEntry(symbol=launch, found=self.resolve(launch) is not None))
        # Probe-only: legacy FIA V1-V4 must be absent on 950PR (EZ9903). We
        # record presence and never bind/invoke them.
        for symbol in LEGACY_FIA_PROBE_SYMBOLS:
            if self.resolve(symbol) is not None:
                report.legacy_fia_present.append(symbol)
        return report

    # ---------------------------------------------------------- descriptors

    def acl_dtype(self, torch_dtype: torch.dtype) -> int:
        _build_torch_dtype_map()
        try:
            return _TORCH_TO_ACL_DTYPE[torch_dtype]
        except KeyError as exc:
            raise ValueError(f"no aclDataType mapping for {torch_dtype}") from exc

    def create_tensor(self, tensor: torch.Tensor) -> AclTensorHandle:
        """Build an ``aclTensor*`` over a contiguous torch tensor's storage.

        Contract: the tensor must be contiguous (slot views and scratchpad
        buffers are narrow+view products, which are). ``data_ptr()`` already
        carries the storage offset, so the descriptor offset stays 0.
        """
        if not tensor.is_contiguous():
            raise ValueError("aclTensor descriptors require contiguous tensors")
        return self.create_tensor_raw(
            tensor.data_ptr(),
            tuple(tensor.shape),
            self.acl_dtype(tensor.dtype),
            strides=tuple(tensor.stride()),
        )

    def create_tensor_raw(
        self,
        data_pointer: int,
        dims: Sequence[int],
        acl_dtype_code: int,
        strides: Sequence[int] | None = None,
    ) -> AclTensorHandle:
        """Descriptor over raw device memory (torch-free bring-up path).

        Packed dtypes (FP4 pairs) and host-filled scale buffers (UE8M0 carried
        in uint8 bytes) pass their ``aclDataType`` code directly; ``dims`` count
        elements, and the memory must hold ``prod(dims) * itemsize`` bytes.
        """
        if strides is None:
            strides = _contiguous_strides(dims)
        # Accept raw ints, c_void_p instances (device buffers) and None (NULL).
        address = getattr(data_pointer, "value", data_pointer)
        view_dims = (ctypes.c_int64 * len(dims))(*dims)
        stride_dims = (ctypes.c_int64 * len(strides))(*strides)
        storage_dims = (ctypes.c_int64 * len(dims))(*dims)
        pointer = self._opapi.aclCreateTensor(
            view_dims,
            len(dims),
            acl_dtype_code,
            stride_dims,
            0,
            ACL_FORMAT_ND,
            storage_dims,
            len(dims),
            ctypes.c_void_p(address),
        )
        if not pointer:
            raise RuntimeError("aclCreateTensor returned nullptr")
        return AclTensorHandle(
            pointer=pointer, view_dims=view_dims, strides=stride_dims, storage_dims=storage_dims
        )

    def create_tensor_list(self, handles: Sequence[AclTensorHandle]) -> AclTensorListHandle:
        """Wrap already-created descriptors into an ``aclTensorList*``."""
        if not handles:
            raise ValueError("aclCreateTensorList requires at least one tensor")
        pointers = (ctypes.c_void_p * len(handles))(*(h.pointer for h in handles))
        pointer = self._opapi.aclCreateTensorList(pointers, len(handles))
        if not pointer:
            raise RuntimeError("aclCreateTensorList returned nullptr")
        return AclTensorListHandle(pointer=pointer, pointers=pointers, handles=list(handles))

    def destroy_tensor_list(self, handle: AclTensorListHandle) -> None:
        if handle.pointer:
            self._opapi.aclDestroyTensorList(handle.pointer)
            handle.pointer = ctypes.c_void_p(0)

    def destroy_tensor(self, handle: AclTensorHandle) -> None:
        if handle.pointer:
            self._opapi.aclDestroyTensor(handle.pointer)
            handle.pointer = ctypes.c_void_p(0)

    def create_int_array(self, values: Sequence[int]) -> AclIntArrayHandle:
        array = (ctypes.c_int64 * len(values))(*values)
        pointer = self._opapi.aclCreateIntArray(array, len(values))
        if not pointer:
            raise RuntimeError("aclCreateIntArray returned nullptr")
        return AclIntArrayHandle(pointer=pointer, values=array)

    def destroy_int_array(self, handle: AclIntArrayHandle) -> None:
        if handle.pointer:
            self._opapi.aclDestroyIntArray(handle.pointer)
            handle.pointer = ctypes.c_void_p(0)

    def raw_ascendcl(self) -> ctypes.CDLL | None:
        return self._ascendcl or self._opapi


if __name__ == "__main__":  # pragma: no cover - manual bring-up probe
    library = AclnnLibrary()
    print(f"libopapi loaded: {library.loaded} ({library.library_path})")
    for entry in library.inventory().entries:
        print(f"  {'found' if entry.found else 'MISSING':7s} {entry.symbol}")
