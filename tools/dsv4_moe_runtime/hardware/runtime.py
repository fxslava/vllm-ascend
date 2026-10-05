"""Hardware-facing runtime seam: one harness runs on npu:0, cuda:0 and cpu.

``DeviceRuntime`` is the only place that knows about ``torch.npu`` vs
``torch.cuda`` vs plain CPU semantics (allocator accounting, copy streams,
cache release, capacity probes); every other module depends on this seam, never
on the hardware directly (DIP). The CUDA runtime exists so the exact same
offload harness can be validated on a workstation GPU (RTX 5070, 12 GiB) before
it runs on Ascend HBM.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

import torch

CUDA_DEVICE_PREFIX = "cuda"
NPU_DEVICE_PREFIX = "npu"
CPU_DEVICE = "cpu"


class DeviceRuntime(Protocol):
    """Hardware-facing seam (allocator accounting, copy stream, sync, capacity)."""

    device: str
    has_allocator_accounting: bool
    supports_pinned_host_memory: bool

    def memory_allocated(self) -> int: ...

    def memory_reserved(self) -> int: ...

    def device_total_memory(self) -> int | None:
        """Physical device memory in bytes; ``None`` where unavailable."""

    def device_free_memory(self) -> int | None:
        """Currently free device memory in bytes; ``None`` where unavailable."""

    def synchronize_device(self) -> None: ...

    def make_stream(self) -> object | None: ...

    def current_stream(self) -> object | None:
        """Stream owning the caller's tensor producers and consumers."""

    def stream_context(self, stream: object | None) -> AbstractContextManager[None]: ...

    def synchronize_stream(self, stream: object | None) -> None: ...

    def adopt_current_stream(self, stream: object | None) -> None:
        """Make ``stream`` wait for whatever is already queued on the caller's stream.

        A copy stream that never waits is only safe if nothing on the default
        stream touches the same memory -- and something does: the slot arena is
        zeroed at construction, on the default stream. Issuing an H2D copy on an
        unsynchronized side stream races that fill, and the losing order leaves
        the destination zeroed with every byte count reporting success.
        """

    def release_cache(self) -> None:
        """Best-effort release of cached allocator blocks (used after OOM)."""

    def reset_device(self) -> None:
        """Post-teardown device cleanup (allocator stats reset; no-op on cpu)."""


class NpuRuntime:
    """Ascend 950PR runtime: torch_npu allocator stats + dedicated DMA stream."""

    def __init__(self, device: str):
        try:
            import torch_npu  # noqa: F401  (registers the torch.npu namespace)
        except ImportError as exc:
            raise RuntimeError(
                f"device {device!r} requires torch_npu (CANN 8.x/9.x, aclnn V5); "
                "run on the Ascend host, or use --device cpu / --dry-run on a workstation"
            ) from exc
        self._device_index = _device_index(device)
        torch.npu.set_device(self._device_index)
        self.device = device
        self.has_allocator_accounting = True
        self.supports_pinned_host_memory = True

    def memory_allocated(self) -> int:
        return int(torch.npu.memory_allocated(self.device))

    def memory_reserved(self) -> int:
        return int(torch.npu.memory_reserved(self.device))

    def device_total_memory(self) -> int | None:
        return _capacity_from(torch.npu, self._device_index)[1]

    def device_free_memory(self) -> int | None:
        return _capacity_from(torch.npu, self._device_index)[0]

    def synchronize_device(self) -> None:
        torch.npu.synchronize(self.device)

    def make_stream(self) -> object:
        return torch.npu.Stream()

    def current_stream(self) -> object:
        return torch.npu.current_stream(self.device)

    def stream_context(self, stream: object | None) -> AbstractContextManager[None]:
        return torch.npu.stream(stream)

    def synchronize_stream(self, stream: object | None) -> None:
        if stream is not None:
            stream.synchronize()

    def adopt_current_stream(self, stream: object | None) -> None:
        if stream is not None:
            stream.wait_stream(torch.npu.current_stream(self.device))

    def release_cache(self) -> None:
        torch.npu.empty_cache()

    def reset_device(self) -> None:
        # Final device pass after every owner is gone: flush the cache again
        # and clear accumulated allocator statistics. A hard aclrtResetDevice
        # would tear the device out from under torch_npu's runtime state, so
        # the lifecycle manager stops at these public entry points.
        torch.npu.empty_cache()
        if hasattr(torch.npu, "reset_accumulated_memory_stats"):
            torch.npu.reset_accumulated_memory_stats(self.device)


class CudaRuntime:
    """RTX 5070 / CUDA runtime: caching-allocator stats + dedicated copy stream.

    The hardware-identical validation target for the Ascend path: allocator
    accounting carries the zero-allocation invariant exactly as ``torch.npu``
    does, pinned host staging comes from ``cudaHostAlloc`` (page-aligned, which
    the unbuffered DirectStorage reads require) and H2D DMA rides a dedicated
    ``torch.cuda.Stream``.
    """

    def __init__(self, device: str):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {device!r} requires a CUDA build of torch with a visible GPU; "
                "use --device cpu / --dry-run on a machine without one"
            )
        self._device_index = _device_index(device)
        device_count = torch.cuda.device_count()
        if not 0 <= self._device_index < device_count:
            raise ValueError(f"{device!r}: CUDA device index out of range (visible devices: {device_count})")
        torch.cuda.set_device(self._device_index)
        self.device = f"{CUDA_DEVICE_PREFIX}:{self._device_index}"
        self.has_allocator_accounting = True
        self.supports_pinned_host_memory = True

    @property
    def device_name(self) -> str:
        return torch.cuda.get_device_name(self._device_index)

    def memory_allocated(self) -> int:
        return int(torch.cuda.memory_allocated(self.device))

    def memory_reserved(self) -> int:
        return int(torch.cuda.memory_reserved(self.device))

    def device_total_memory(self) -> int | None:
        return _capacity_from(torch.cuda, self._device_index)[1]

    def device_free_memory(self) -> int | None:
        return _capacity_from(torch.cuda, self._device_index)[0]

    def synchronize_device(self) -> None:
        torch.cuda.synchronize(self.device)

    def make_stream(self) -> object:
        return torch.cuda.Stream(device=self.device)

    def current_stream(self) -> object:
        return torch.cuda.current_stream(self.device)

    def stream_context(self, stream: object | None) -> AbstractContextManager[None]:
        return torch.cuda.stream(stream)

    def synchronize_stream(self, stream: object | None) -> None:
        if stream is not None:
            stream.synchronize()

    def adopt_current_stream(self, stream: object | None) -> None:
        if stream is not None:
            stream.wait_stream(torch.cuda.current_stream(self.device))

    def release_cache(self) -> None:
        torch.cuda.empty_cache()
        # Cached page-locked blocks still count against physical host memory.
        # Release unused blocks between complete hierarchy benchmark instances.
        empty_host_cache = getattr(getattr(torch, "accelerator", None), "empty_host_cache", None)
        if empty_host_cache is not None:
            empty_host_cache()

    def reset_device(self) -> None:
        # Mirrors NpuRuntime: flush the cache and clear accumulated allocator
        # statistics. A hard device reset would invalidate torch's own CUDA
        # context, so teardown stops at these public entry points.
        torch.cuda.empty_cache()
        torch.cuda.reset_accumulated_memory_stats(self._device_index)


class CpuRuntime:
    """Workstation runtime: no device allocator accounting, no copy stream."""

    def __init__(self, device: str = CPU_DEVICE):
        if device != CPU_DEVICE:
            raise ValueError(f"CpuRuntime serves only 'cpu', got {device!r}")
        self.device = CPU_DEVICE
        self.has_allocator_accounting = False
        self.supports_pinned_host_memory = torch.cuda.is_available()

    def memory_allocated(self) -> int:
        return 0

    def memory_reserved(self) -> int:
        return 0

    def device_total_memory(self) -> int | None:
        return None

    def device_free_memory(self) -> int | None:
        return None

    def synchronize_device(self) -> None:
        pass

    def make_stream(self) -> object | None:
        return None

    def current_stream(self) -> object | None:
        return None

    def stream_context(self, stream: object | None) -> AbstractContextManager[None]:
        return nullcontext()

    def synchronize_stream(self, stream: object | None) -> None:
        pass

    def adopt_current_stream(self, stream: object | None) -> None:
        pass

    def release_cache(self) -> None:
        pass

    def reset_device(self) -> None:
        pass


def _device_index(device: str) -> int:
    """Index of ``<backend>[:<index>]``; a bare backend name means device 0."""
    _backend, _, index = device.partition(":")
    if not index:
        return 0
    if not index.isdigit():
        raise ValueError(f"device {device!r}: expected '<backend>:<index>'")
    return int(index)


def _capacity_from(namespace: object, index: int) -> tuple[int | None, int | None]:
    """``(free, total)`` device bytes via ``mem_get_info``, with a total-only fallback."""
    mem_get_info = getattr(namespace, "mem_get_info", None)
    if mem_get_info is not None:
        try:
            free, total = mem_get_info(index)
            return int(free), int(total)
        except (RuntimeError, AssertionError, TypeError):  # older/partial backends
            pass
    properties = getattr(namespace, "get_device_properties", None)
    if properties is None:
        return None, None
    try:
        return None, int(properties(index).total_memory)
    except (RuntimeError, AssertionError, AttributeError):
        return None, None


def make_runtime(device: str) -> DeviceRuntime:
    """Build the runtime seam for ``cpu``, ``cuda[:i]`` or ``npu[:i]``."""
    backend = device.partition(":")[0]
    if backend == CPU_DEVICE:
        return CpuRuntime()
    if backend == CUDA_DEVICE_PREFIX:
        return CudaRuntime(device)
    if backend == NPU_DEVICE_PREFIX:
        return NpuRuntime(device)
    raise ValueError(f"unsupported device {device!r}; expected cpu, cuda[:i] or npu[:i]")
