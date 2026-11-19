# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional dispatch logging and inference profiling, independent of attention.

Execution implementation (PyTorch or a custom operator) and tensor device are
separate facts. Profilers are selected by device, so PyTorch operations on an
NPU use the NPU profiler while CPU execution uses torch.profiler. Additional
devices can supply a ProfilerBackend without changing the inference wrapper.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from enum import Enum
from types import TracebackType
from typing import Any, Final, Protocol, TypeVar

import torch
from vllm.logger import logger

from vllm_ascend import envs

TURBOQUANT_PROFILE_ENV: Final[str] = "TURBOQUANT_PROFILE"
TURBOQUANT_PROFILE_TRACE_DIR: Final[str] = "./turboquant_traces"
TURBOQUANT_PROFILE_NOTICE: Final[str] = (
    "TurboQuant profiling active on %s. Traces are written to %s for offline analysis."
)

_Result = TypeVar("_Result")


def turboquant_profile_enabled() -> bool:
    return envs.TURBOQUANT_PROFILE


class DispatchImplementation(str, Enum):
    PYTORCH = "pytorch"
    CUSTOM_OP = "custom_op"


class DispatchObserver(Protocol):
    def record(
        self, operation: str, implementation: DispatchImplementation, **tensors: torch.Tensor | None
    ) -> None: ...


class NullDispatchObserver:
    def record(self, operation: str, implementation: DispatchImplementation, **tensors: torch.Tensor | None) -> None:
        pass


class LoggingDispatchObserver:
    """Read tensor metadata only; never copy or synchronise tensor contents."""

    def record(self, operation: str, implementation: DispatchImplementation, **tensors: torch.Tensor | None) -> None:
        tensor_items = [(name, tensor) for name, tensor in tensors.items() if isinstance(tensor, torch.Tensor)]
        devices = sorted({str(tensor.device) for _, tensor in tensor_items})
        layouts = "; ".join(
            f"{name}(shape={tuple(tensor.shape)}, stride={tensor.stride()}, dtype={tensor.dtype}, "
            f"device={tensor.device}, storage_offset={tensor.storage_offset()}, "
            f"contiguous={tensor.is_contiguous()})"
            for name, tensor in tensor_items
        )
        logger.info(
            "TurboQuant dispatch: op=%s implementation=%s devices=%s tensors=[%s]",
            operation,
            implementation.value,
            devices or "none",
            layouts,
        )


def create_dispatch_observer() -> DispatchObserver:
    return LoggingDispatchObserver() if envs.VLLM_ASCEND_TURBOQUANT_TRACE else NullDispatchObserver()


class ProfilerSession(Protocol):
    def __enter__(self) -> ProfilerSession: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> object: ...

    def step(self) -> None: ...


class ProfilerBackend(Protocol):
    def is_capturing(self) -> bool: ...

    def open_session(self, trace_directory: str) -> ProfilerSession: ...


class TorchProfilerBackend:
    """CPU trace export through the standard PyTorch profiler."""

    def is_capturing(self) -> bool:
        return False

    def open_session(self, trace_directory: str) -> ProfilerSession:
        return torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_directory),
            record_shapes=False,
            with_stack=False,
        )


class NpuProfilerBackend:
    """Keep torch_npu optional until an NPU trace is explicitly requested."""

    def is_capturing(self) -> bool:
        # Missing capture support is common in older torch_npu builds. An
        # actual runtime failure is handled by InferenceTelemetry, which skips
        # measurement rather than assuming that opening a profiler is safe.
        capture_query = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
        return bool(capture_query()) if capture_query is not None else False

    def open_session(self, trace_directory: str) -> ProfilerSession:
        profiler = importlib.import_module("torch_npu").profiler
        return profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
            on_trace_ready=profiler.tensorboard_trace_handler(trace_directory),
            record_shapes=False,
            with_stack=False,
        )


def create_profiler_backend(device_type: str) -> ProfilerBackend:
    if device_type == "cpu":
        return TorchProfilerBackend()
    if device_type == "npu":
        return NpuProfilerBackend()
    raise ValueError(f"TurboQuant profiling does not support device {device_type!r}")


class InferenceTelemetry:
    """Measure a call exactly once, isolating profiler failures from inference.

    A backend factory can be injected for another device or a test. Disabled
    telemetry does not construct a backend, inspect a stream, or create files.
    A profiler failure disables this layer's telemetry for subsequent calls.
    """

    def __init__(
        self,
        enabled: bool,
        *,
        backend_factory: Callable[[str], ProfilerBackend] = create_profiler_backend,
        trace_directory: str = TURBOQUANT_PROFILE_TRACE_DIR,
    ) -> None:
        self.enabled = enabled
        self._backend_factory = backend_factory
        self._trace_directory = trace_directory
        self._backends: dict[str, ProfilerBackend] = {}

    def run(self, forward: Callable[..., _Result], device_type: str, **arguments: Any) -> _Result:
        if not self.enabled:
            return forward(**arguments)
        try:
            backend = self._backends.get(device_type)
            if backend is None:
                backend = self._backend_factory(device_type)
                self._backends[device_type] = backend
            capturing = backend.is_capturing()
        except Exception as error:
            self._disable(error)
            return forward(**arguments)
        if capturing:
            return forward(**arguments)
        try:
            session = backend.open_session(self._trace_directory)
            active_session = session.__enter__()
        except Exception as error:
            self._disable(error)
            return forward(**arguments)

        logger.info_once(TURBOQUANT_PROFILE_NOTICE, device_type, self._trace_directory)
        try:
            result = forward(**arguments)
        except BaseException as error:
            # A profiler must never suppress or replace an inference exception.
            self._close_session(session, type(error), error, error.__traceback__)
            raise
        else:
            try:
                active_session.step()
            except Exception as error:
                self._disable(error)
            self._close_session(session, None, None, None)
            return result

    def _close_session(
        self,
        session: ProfilerSession,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            session.__exit__(exc_type, exc_value, traceback)
        except Exception as error:
            self._disable(error)

    def _disable(self, reason: object) -> None:
        self.enabled = False
        logger.warning("TurboQuant profiling disabled for this layer; inference continues: %s", reason)
