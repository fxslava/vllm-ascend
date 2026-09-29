"""Hardware-facing runtime seam: the identical harness runs on npu:0 and cpu.

``DeviceRuntime`` is the only place that knows about ``torch.npu`` vs plain
CPU semantics (allocator accounting, copy streams, cache release); every other
module depends on this seam, never on the hardware directly (DIP).
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Protocol

import torch


class DeviceRuntime(Protocol):
    """Hardware-facing seam (allocator accounting, copy stream, sync)."""

    device: str
    has_allocator_accounting: bool
    supports_pinned_host_memory: bool

    def memory_allocated(self) -> int: ...

    def memory_reserved(self) -> int: ...

    def synchronize_device(self) -> None: ...

    def make_stream(self) -> object | None: ...

    def stream_context(self, stream: object | None) -> AbstractContextManager[None]: ...

    def synchronize_stream(self, stream: object | None) -> None: ...

    def release_cache(self) -> None:
        """Best-effort release of cached allocator blocks (used after OOM)."""


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
        torch.npu.set_device(int(device.rsplit(":", 1)[-1]))
        self.device = device
        self.has_allocator_accounting = True
        self.supports_pinned_host_memory = True

    def memory_allocated(self) -> int:
        return int(torch.npu.memory_allocated(self.device))

    def memory_reserved(self) -> int:
        return int(torch.npu.memory_reserved(self.device))

    def synchronize_device(self) -> None:
        torch.npu.synchronize(self.device)

    def make_stream(self) -> object:
        return torch.npu.Stream()

    def stream_context(self, stream: object | None) -> AbstractContextManager[None]:
        return torch.npu.stream(stream)

    def synchronize_stream(self, stream: object | None) -> None:
        if stream is not None:
            stream.synchronize()

    def release_cache(self) -> None:
        torch.npu.empty_cache()


class CpuRuntime:
    """Workstation runtime: no device allocator accounting, no copy stream."""

    def __init__(self, device: str = "cpu"):
        if device != "cpu":
            raise ValueError(f"CpuRuntime serves only 'cpu', got {device!r}")
        self.device = "cpu"
        self.has_allocator_accounting = False
        self.supports_pinned_host_memory = torch.cuda.is_available()

    def memory_allocated(self) -> int:
        return 0

    def memory_reserved(self) -> int:
        return 0

    def synchronize_device(self) -> None:
        pass

    def make_stream(self) -> object | None:
        return None

    def stream_context(self, stream: object | None) -> AbstractContextManager[None]:
        return nullcontext()

    def synchronize_stream(self, stream: object | None) -> None:
        pass

    def release_cache(self) -> None:
        pass


def make_runtime(device: str) -> DeviceRuntime:
    if device == "cpu":
        return CpuRuntime()
    return NpuRuntime(device)
