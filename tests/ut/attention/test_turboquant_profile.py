# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the TurboQuant dry run's in-code profiler.

The profiler exports a trace; it does not summarise one. torch_npu has no
in-memory aggregation -- ``prof.key_averages()`` does not exist on it -- so
everything readable comes out of the handler's files, and what is left to test
on a host with no NPU, no CANN and no profiler is the plumbing: that a profiled
forward is configured to write, that it runs exactly once, and that the
measurement cannot change what the forward returns or raises.

What is pinned:

* the profiler is opened with ``on_trace_ready``, pointed at
  :data:`TURBOQUANT_PROFILE_TRACE_DIR`, and with both activities. A profiler
  without a handler would record a step that nothing can ever read.
* ``prof.step()`` is called inside the context, which is what makes the handler
  chunk and flush per step rather than once at the end of the run.
* the notice naming the trace directory is logged through ``info_once``, on the
  first profiled forward.
* a build whose ``torch_npu.profiler`` is missing, or missing the trace
  handler, leaves the forward alone and switches profiling off.
* the env gate is off unless asked.
"""

from __future__ import annotations

import types

import pytest

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.turboquant_dry_run import (
    TURBOQUANT_PROFILE_ENV,
    TURBOQUANT_PROFILE_NOTICE,
    TURBOQUANT_PROFILE_TRACE_DIR,
    AscendTurboQuantDryRunBackendImpl,
    turboquant_profile_enabled,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), ("0", False), ("", False), ("1", True), ("true", True), ("ON", True), ("2", False)],
)
def test_the_env_gate_is_off_unless_asked(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(TURBOQUANT_PROFILE_ENV, raising=False)
    else:
        monkeypatch.setenv(TURBOQUANT_PROFILE_ENV, value)
    assert turboquant_profile_enabled() is expected


def test_the_notice_names_the_trace_directory_and_takes_it_as_an_argument():
    """``info_once`` caches on its arguments, so the directory is a ``%s``."""
    assert TURBOQUANT_PROFILE_NOTICE.count("%s") == 1
    assert "key_averages" in TURBOQUANT_PROFILE_NOTICE
    assert (TURBOQUANT_PROFILE_NOTICE % TURBOQUANT_PROFILE_TRACE_DIR).find("./turboquant_traces") > 0


# -- the plumbing around the profiler ----------------------------------------
#
# A fake torch_npu.profiler stands in for the real one, which needs a device.


class FakeProf:
    """As much of ``torch_npu.profiler.profile``'s context as is used."""

    def __init__(self):
        self.entered = False
        self.exited = False
        self.steps_inside_context = 0

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc_info):
        self.exited = True
        return False

    def step(self):
        # A handler chunks on step(); stepping a stopped profiler would not.
        assert self.entered and not self.exited, "step() outside the context"
        self.steps_inside_context += 1


@pytest.fixture
def profiled(monkeypatch):
    """An impl with profiling on and a fake torch_npu.profiler behind it."""
    import vllm_ascend.attention.turboquant_dry_run as module

    state: dict[str, object] = {"calls": 0, "notices": [], "handlers": []}

    def tensorboard_trace_handler(directory):
        state["handlers"] = [*state["handlers"], directory]  # type: ignore[misc]
        return f"handler({directory})"

    def profile(**kwargs):
        state["kwargs"] = kwargs
        state["prof"] = FakeProf()
        return state["prof"]

    monkeypatch.setattr(
        module,
        "torch_npu",
        types.SimpleNamespace(
            profiler=types.SimpleNamespace(
                ProfilerActivity=types.SimpleNamespace(CPU="cpu", NPU="npu"),
                profile=profile,
                tensorboard_trace_handler=tensorboard_trace_handler,
            )
        ),
    )
    monkeypatch.setattr(
        module,
        "logger",
        types.SimpleNamespace(
            info_once=lambda template, *args: state["notices"].append(template % args),  # type: ignore[union-attr]
            info=lambda *args: None,
            warning=lambda *args: None,
        ),
    )

    impl = object.__new__(AscendTurboQuantDryRunBackendImpl)
    impl._profile = True

    def base_forward(self, **call):
        state["calls"] = int(state["calls"]) + 1
        if state.get("raises"):
            raise ValueError("the forward's own failure")
        return "attn_output"

    monkeypatch.setattr(module.AscendAttentionBackendImpl, "forward", base_forward, raising=False)
    state["impl"] = impl
    state["module"] = module
    return state


def _run(profiled, **call):
    return profiled["impl"]._profiled_forward(dict(call))


def test_a_profiled_forward_runs_once_and_is_configured_to_export(profiled):
    assert _run(profiled) == "attn_output"
    assert profiled["calls"] == 1

    kwargs = profiled["kwargs"]
    assert kwargs["activities"] == ["cpu", "npu"]
    # The handler is the whole point: without it the step is unreadable.
    assert kwargs["on_trace_ready"] == f"handler({TURBOQUANT_PROFILE_TRACE_DIR})"
    assert profiled["handlers"] == [TURBOQUANT_PROFILE_TRACE_DIR]
    # Nothing is read back, so neither of the costly record levels is asked for.
    assert kwargs["record_shapes"] is False
    assert kwargs["with_stack"] is False

    prof = profiled["prof"]
    assert prof.entered and prof.exited
    assert prof.steps_inside_context == 1


def test_the_notice_is_logged_once_however_many_forwards_are_profiled(profiled):
    for _ in range(3):
        assert _run(profiled) == "attn_output"

    # The impl calls info_once; the real logger dedupes on the arguments, and
    # the fake records every call, so what is pinned here is the single
    # template and its single argument.
    assert profiled["notices"] == [TURBOQUANT_PROFILE_NOTICE % TURBOQUANT_PROFILE_TRACE_DIR] * 3
    assert "./turboquant_traces" in profiled["notices"][0]
    assert profiled["calls"] == 3


def test_a_profiler_that_cannot_be_constructed_leaves_the_forward_alone(profiled):
    def refuse(**kwargs):
        raise TypeError("profile() got an unexpected keyword argument 'on_trace_ready'")

    profiled["module"].torch_npu.profiler.profile = refuse

    assert _run(profiled) == "attn_output"
    assert profiled["calls"] == 1
    assert profiled["impl"]._profile is False
    assert profiled["notices"] == []  # nothing is being written, so nothing is announced


@pytest.mark.parametrize("missing", ["profiler", "ProfilerActivity", "tensorboard_trace_handler"])
def test_an_unavailable_profiler_leaves_the_forward_alone(profiled, missing):
    module = profiled["module"]
    if missing == "profiler":
        module.torch_npu = types.SimpleNamespace(profiler=None)
    else:
        setattr(module.torch_npu.profiler, missing, None)

    assert _run(profiled) == "attn_output"
    assert profiled["calls"] == 1
    assert profiled["impl"]._profile is False
    assert profiled["notices"] == []


def test_the_forwards_own_failure_is_not_reported_as_a_profiler_failure(profiled):
    profiled["raises"] = True

    with pytest.raises(ValueError, match="the forward's own failure"):
        _run(profiled)
    # Still on: nothing about the profiler went wrong.
    assert profiled["impl"]._profile is True


def test_capture_and_a_metadataless_forward_are_never_profiled(monkeypatch, profiled):
    """The two cases ``forward`` keeps away from the profiler, by its own gate."""
    module = profiled["module"]
    impl = profiled["impl"]
    opened: list[int] = []
    monkeypatch.setattr(
        AscendTurboQuantDryRunBackendImpl,
        "_open_profiler",
        lambda self: opened.append(1),  # type: ignore[arg-type,return-value]
    )
    monkeypatch.setattr(impl, "_ensure_statics", lambda layer, device: None, raising=False)

    layer = type("Layer", (), {"layer_name": "model.layers.3.self_attn.attn"})()
    metadata = type("Metadata", (), {"attn_state": AscendAttentionState.DecodeOnly})()
    query = types.SimpleNamespace(device="npu")

    def run(attn_metadata, capturing):
        monkeypatch.setattr(module, "_stream_is_capturing", lambda: capturing)
        return AscendTurboQuantDryRunBackendImpl.forward(
            impl,
            layer=layer,
            query=query,
            key=None,
            value=None,
            kv_cache=(),
            attn_metadata=attn_metadata,
        )

    assert run(None, False) == "attn_output"  # a warm-up forward
    assert run(metadata, True) == "attn_output"  # inside an ACL graph capture
    assert opened == []  # no profiler was ever opened
    assert profiled["calls"] == 2
