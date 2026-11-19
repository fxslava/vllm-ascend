# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-free contracts for profiler selection and inference isolation."""

from __future__ import annotations

import inspect
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend import envs
from vllm_ascend.attention import turboquant_ops, turboquant_v1
from vllm_ascend.attention import turboquant_telemetry as telemetry_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.turboquant_telemetry import (
    TURBOQUANT_PROFILE_TRACE_DIR,
    InferenceTelemetry,
    NpuProfilerBackend,
    TorchProfilerBackend,
    create_profiler_backend,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("0", False),
        ("", False),
        ("1", True),
        ("true", True),
        (" ON ", True),
        ("yes", True),
        ("2", False),
    ],
)
def test_profile_environment_gate(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("TURBOQUANT_PROFILE", raising=False)
    else:
        monkeypatch.setenv("TURBOQUANT_PROFILE", value)
    assert envs.TURBOQUANT_PROFILE is expected


class FakeSession:
    def __init__(self, failure=None, suppress=False):
        self.failure = failure
        self.suppress = suppress
        self.entered = False
        self.exited = False
        self.steps = 0
        self.exception = None

    def __enter__(self):
        if self.failure == "enter":
            raise RuntimeError("enter failed")
        self.entered = True
        self.exited = False
        return self

    def step(self):
        assert self.entered and not self.exited
        self.steps += 1
        if self.failure == "step":
            raise RuntimeError("step failed")

    def __exit__(self, exc_type, exc_value, traceback):
        self.exited = True
        self.exception = exc_value
        if self.failure == "exit":
            raise RuntimeError("exit failed")
        return self.suppress


@pytest.fixture
def measurement(monkeypatch):
    session = FakeSession()
    backend = Mock()
    backend.is_capturing.return_value = False
    backend.open_session.return_value = session
    factory = Mock(return_value=backend)
    logger = Mock()
    monkeypatch.setattr(telemetry_module, "logger", logger)
    telemetry = InferenceTelemetry(True, backend_factory=factory)
    return SimpleNamespace(session=session, backend=backend, factory=factory, logger=logger, telemetry=telemetry)


def test_measurement_preserves_result_and_steps_inside_context(measurement):
    result = object()
    forward = Mock(return_value=result)
    assert measurement.telemetry.run(forward, "cpu", query="query") is result
    forward.assert_called_once_with(query="query")
    measurement.factory.assert_called_once_with("cpu")
    measurement.backend.open_session.assert_called_once_with(TURBOQUANT_PROFILE_TRACE_DIR)
    assert measurement.session.entered and measurement.session.exited
    assert measurement.session.steps == 1
    measurement.logger.info_once.assert_called_once_with(
        telemetry_module.TURBOQUANT_PROFILE_NOTICE, "cpu", TURBOQUANT_PROFILE_TRACE_DIR
    )


def test_disabled_telemetry_does_not_construct_backend(measurement):
    measurement.telemetry.enabled = False
    forward = Mock(return_value="output")
    assert measurement.telemetry.run(forward, "npu") == "output"
    measurement.factory.assert_not_called()
    measurement.logger.info_once.assert_not_called()


@pytest.mark.parametrize("inference_fails", [False, True])
def test_capture_never_opens_profiler_or_repeats_inference(measurement, inference_fails):
    measurement.backend.is_capturing.return_value = True
    failure = ValueError("inference failed")
    forward = Mock(side_effect=failure) if inference_fails else Mock(return_value="output")
    if inference_fails:
        with pytest.raises(ValueError) as caught:
            measurement.telemetry.run(forward, "npu")
        assert caught.value is failure
    else:
        assert measurement.telemetry.run(forward, "npu") == "output"
    forward.assert_called_once_with()
    measurement.backend.open_session.assert_not_called()
    assert measurement.telemetry.enabled


@pytest.mark.parametrize("failure", ["factory", "capture", "open", "enter", "step", "exit"])
def test_profiler_failures_disable_measurement_without_repeating_inference(measurement, failure):
    if failure == "factory":
        measurement.factory.side_effect = RuntimeError("factory failed")
    elif failure == "capture":
        measurement.backend.is_capturing.side_effect = RuntimeError("capture failed")
    elif failure == "open":
        measurement.backend.open_session.side_effect = RuntimeError("open failed")
    else:
        measurement.session.failure = failure
    forward = Mock(return_value="output")
    assert measurement.telemetry.run(forward, "npu") == "output"
    forward.assert_called_once_with()
    assert not measurement.telemetry.enabled
    assert measurement.telemetry.run(forward, "npu") == "output"
    assert forward.call_count == 2
    assert measurement.factory.call_count == 1
    measurement.logger.warning.assert_called()
    if failure == "step":
        assert measurement.session.exited


@pytest.mark.parametrize("exit_failure", [False, True])
@pytest.mark.parametrize("suppress", [False, True])
def test_inference_exception_is_preserved_even_if_profiler_suppresses_or_raises(measurement, exit_failure, suppress):
    measurement.session.failure = "exit" if exit_failure else None
    measurement.session.suppress = suppress
    failure = ValueError("the inference failure")
    forward = Mock(side_effect=failure)
    with pytest.raises(ValueError) as caught:
        measurement.telemetry.run(forward, "cpu")
    assert caught.value is failure
    assert measurement.session.exception is failure
    assert measurement.session.steps == 0
    forward.assert_called_once_with()
    assert measurement.telemetry.enabled is not exit_failure


def test_backend_factory_is_cached_per_device_and_extensible(measurement):
    for device in ("cpu", "npu", "future_device", "cpu"):
        measurement.telemetry.run(lambda: None, device)
    assert [call.args for call in measurement.factory.call_args_list] == [("cpu",), ("npu",), ("future_device",)]


def test_default_backend_selection():
    assert isinstance(create_profiler_backend("cpu"), TorchProfilerBackend)
    assert isinstance(create_profiler_backend("npu"), NpuProfilerBackend)
    with pytest.raises(ValueError, match="does not support device"):
        create_profiler_backend("unsupported")


@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_backend_configures_only_its_supported_activities(monkeypatch, device):
    profiler = Mock()
    profiler.ProfilerActivity = SimpleNamespace(CPU="cpu", NPU="npu")
    if device == "cpu":
        monkeypatch.setattr(torch, "profiler", profiler)
        # CPU profiling must work even when torch_npu cannot be imported.
        monkeypatch.setitem(sys.modules, "torch_npu", None)
    else:
        monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=profiler))
    backend = create_profiler_backend(device)
    assert backend.open_session("traces") is profiler.profile.return_value
    profiler.tensorboard_trace_handler.assert_called_once_with("traces")
    profiler.profile.assert_called_once_with(
        activities=["cpu"] if device == "cpu" else ["cpu", "npu"],
        on_trace_ready=profiler.tensorboard_trace_handler.return_value,
        record_shapes=False,
        with_stack=False,
    )


@pytest.mark.parametrize("missing", ["profiler", "ProfilerActivity", "tensorboard_trace_handler", "profile", "NPU"])
def test_unavailable_npu_profiler_leaves_inference_running(monkeypatch, measurement, missing):
    profiler = SimpleNamespace(
        ProfilerActivity=SimpleNamespace(CPU="cpu", NPU="npu"),
        tensorboard_trace_handler=Mock(),
        profile=Mock(),
    )
    if missing == "profiler":
        module = SimpleNamespace()
    else:
        module = SimpleNamespace(profiler=profiler)
        delattr(profiler.ProfilerActivity if missing == "NPU" else profiler, missing)
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: False), raising=False)
    telemetry = InferenceTelemetry(True)
    forward = Mock(return_value="output")
    assert telemetry.run(forward, "npu") == "output"
    forward.assert_called_once_with()
    assert not telemetry.enabled


def test_npu_capture_query_uses_device_stream(monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_current_stream_capturing=lambda: True), raising=False)
    assert NpuProfilerBackend().is_capturing()
    assert not TorchProfilerBackend().is_capturing()


@pytest.mark.parametrize(
    ("has_metadata", "capturing", "expected_calls"), [(False, False, 0), (True, True, 0), (True, False, 1)]
)
def test_attention_only_delegates_eligible_forwards_to_telemetry(monkeypatch, has_metadata, capturing, expected_calls):
    impl = object.__new__(turboquant_v1.AscendTurboQuantBackendImpl)
    impl.telemetry = Mock()
    impl.telemetry.run.side_effect = lambda forward, device, **arguments: forward(**arguments)
    impl._bind_codec_constants = Mock()
    impl.adapter = SimpleNamespace(has_kernel=lambda name: True)
    impl.emulator = object()
    monkeypatch.setattr(turboquant_v1, "_stream_is_capturing", lambda: capturing)
    base_forward = Mock(return_value="output")
    monkeypatch.setattr(turboquant_v1.AscendAttentionBackendImpl, "forward", base_forward, raising=False)
    metadata = SimpleNamespace(attn_state=AscendAttentionState.DecodeOnly, turboquant_slot_mapping=object())
    query = SimpleNamespace(device=SimpleNamespace(type="npu"))
    assert impl.forward(None, query, None, None, (), metadata if has_metadata else None) == "output"
    assert impl.telemetry.run.call_count == expected_calls
    base_forward.assert_called_once()


def test_debug_dump_paths_remain_absent():
    for module in (turboquant_v1, turboquant_ops, telemetry_module):
        source = inspect.getsource(module)
        for removed in ("TURBOQUANT_DECODE_VALIDATE", "tq_gh6_crash_dump", "tq_camodel_dump", ".tofile("):
            assert removed not in source


def test_public_profile_imports_remain_compatible():
    assert turboquant_v1.turboquant_profile_enabled is telemetry_module.turboquant_profile_enabled
    assert turboquant_v1.TURBOQUANT_PROFILE_ENV == "TURBOQUANT_PROFILE"
    assert turboquant_v1.TURBOQUANT_PROFILE_TRACE_DIR == TURBOQUANT_PROFILE_TRACE_DIR


def test_cpu_profiler_exports_a_readable_trace_without_npu(monkeypatch, tmp_path, measurement):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    telemetry = InferenceTelemetry(True, trace_directory=str(tmp_path))
    inputs = torch.ones(8)
    result = telemetry.run(lambda: inputs + 2, "cpu")
    assert torch.equal(result, torch.full_like(inputs, 3))
    traces = list(tmp_path.glob("*.pt.trace.json"))
    assert traces, "CPU profiling completed without exporting a trace"
    events = json.loads(traces[0].read_text(encoding="utf-8"))["traceEvents"]
    assert any(event.get("name") == "aten::add" for event in events)
