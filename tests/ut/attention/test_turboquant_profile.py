# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the TurboQuant dry run's in-code profiler.

Everything here is pure bookkeeping over profiler rows, so it runs with no NPU,
no CANN and no profiler: the summariser takes anything with a name and the two
``self_*`` totals, which is the contract ``prof.key_averages()`` satisfies.

What is pinned:

* the three buckets are fed from ``self_*`` totals only. Summing the inclusive
  totals instead would charge a child op to its parent's bucket as well, and the
  shares -- the only thing in the log worth reading -- would be meaningless.
* a custom kernel's *device* time is its own, and its *host* time is the host
  bucket's. The host bucket is the cost of asking, over every row.
* the collector logs once per model step, not once per layer, and the boundary
  is a layer label repeating.
* the env gate is off unless asked, and a layer label survives a layer object
  that has only ``layer_name``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.turboquant_dry_run import (
    TURBOQUANT_PROFILE_ENV,
    AscendTurboQuantDryRunBackendImpl,
    TurboQuantProfileLog,
    summarise_turboquant_profile,
    turboquant_profile_enabled,
)


@dataclass
class Row:
    """A ``FunctionEventAvg``, as much of one as the summariser reads."""

    key: str
    self_cpu_time_total: float = 0.0
    self_npu_time_total: float = 0.0


def _summarise(rows, **kwargs):
    defaults = {"layer": "0", "phase": "Prefill", "host_wall_micros": 1000.0}
    return summarise_turboquant_profile(rows, **{**defaults, **kwargs})


# A step's worth of rows: one custom writer, one custom decode, two dense ops,
# and a host-only row that never reached the device.
ROWS = [
    Row("turboquant_ops::npu_turboquant_reshape_and_cache_a8", self_cpu_time_total=300.0, self_npu_time_total=6000.0),
    Row("npu_turboquant_fused_decode", self_cpu_time_total=200.0, self_npu_time_total=2000.0),
    Row("npu_fused_infer_attention_score_v2", self_cpu_time_total=250.0, self_npu_time_total=1500.0),
    Row("aten::index_select", self_cpu_time_total=150.0, self_npu_time_total=500.0),
    Row("aten::empty", self_cpu_time_total=100.0),
]


def test_buckets_split_custom_from_native_and_sum_host_over_every_row():
    profile = _summarise(ROWS)
    custom, native, host = profile.buckets

    assert (custom.label, native.label, host.label) == ("Custom Kernels", "PyTorch Native", "CPU/Host")
    # Both turboquant rows, by device time, and nothing else.
    assert custom.micros == pytest.approx(8000.0)
    assert custom.top_ops[0] == "turboquant_ops::npu_turboquant_reshape_and_cache_a8"
    assert all("turboquant" in name for name in custom.top_ops)
    # The dense ops, and not the host-only row, which has no device time.
    assert native.micros == pytest.approx(2000.0)
    assert set(native.top_ops) == {"npu_fused_infer_attention_score_v2", "aten::index_select"}
    # Host time over every row, the custom kernels' launches included.
    assert host.micros == pytest.approx(1000.0)
    assert profile.tdot_micros == pytest.approx(11000.0)


def test_shares_are_of_tdot_and_the_wall_is_reported_beside_it():
    profile = _summarise(ROWS)
    rendered = profile.render()

    header = "[TurboQuant Profile] Layer 0 (Prefill) | TDOT: 11.000ms (host wall 1.000ms)"
    assert rendered.splitlines()[0] == header
    # 8/11, 2/11, 1/11 of TDOT -- not of the wall, which is smaller than any
    # of them because the device work outlives the launch that queued it.
    assert " ├─ Custom Kernels : 8.000ms (72.7%)" in rendered
    assert " ├─ PyTorch Native : 2.000ms (18.2%)" in rendered
    assert " └─ CPU/Host       : 1.000ms (9.1%)" in rendered


def test_inclusive_totals_are_never_read():
    """A parent row carrying a child's time must not be counted twice."""
    parent = Row("npu_turboquant_fused_decode", self_cpu_time_total=10.0, self_npu_time_total=100.0)
    # The spellings key_averages also offers, all of them inclusive.
    parent.npu_time_total = 900.0  # type: ignore[attr-defined]
    parent.cpu_time_total = 500.0  # type: ignore[attr-defined]
    parent.self_npu_time = 42.0  # type: ignore[attr-defined]

    custom, _, host = _summarise([parent]).buckets
    assert custom.micros == pytest.approx(100.0)
    assert host.micros == pytest.approx(10.0)


def test_device_time_under_the_torch_spelling_is_still_device_time():
    """A torch_npu that reports ``self_device_time_total`` is not a zero column."""

    @dataclass
    class DeviceRow:
        key: str
        self_cpu_time_total: float = 0.0
        self_device_time_total: float = 0.0

    custom, native, _ = _summarise(
        [
            DeviceRow("npu_turboquant_fused_decode", self_device_time_total=70.0),
            DeviceRow("aten::cat", self_device_time_total=30.0),
        ]
    ).buckets
    assert (custom.micros, native.micros) == (pytest.approx(70.0), pytest.approx(30.0))


def test_top_ops_are_ranked_and_truncation_is_announced():
    rows = [Row(f"aten::op_{index}", self_npu_time_total=float(index)) for index in range(6)]
    _, native, _ = _summarise(rows, top_ops=2).buckets

    assert native.top_ops == ("aten::op_5", "aten::op_4")
    assert native.truncated
    assert native.render(native.micros, "├─").endswith("[aten::op_5, aten::op_4, ...]")


def test_an_empty_profile_renders_without_dividing_by_zero():
    profile = _summarise([], host_wall_micros=0.0)
    assert profile.tdot_micros == 0.0
    rows = profile.render().splitlines()[1:]
    assert len(rows) == 3
    assert all(row.endswith("(0.0%)") for row in rows)  # no share, and no op list


def test_the_collector_logs_once_per_step_and_flushes_on_a_repeated_layer(monkeypatch):
    records: list[str] = []
    collector = TurboQuantProfileLog()

    import vllm_ascend.attention.turboquant_dry_run as module

    class Capture:
        def info(self, template, body):
            records.append(template % body)

    monkeypatch.setattr(module, "logger", Capture())

    for layer in ("0", "1", "2"):
        collector.record(_summarise(ROWS, layer=layer))
    assert records == []  # the step is not over until a label comes round again

    collector.record(_summarise(ROWS, layer="0"))
    assert len(records) == 1
    step = records[0]
    assert step.startswith("[TurboQuant Profile] step: 3 layer(s), TDOT 33.000ms")
    assert [line for line in step.splitlines() if line.startswith("[TurboQuant Profile] Layer")] == [
        "[TurboQuant Profile] Layer 0 (Prefill) | TDOT: 11.000ms (host wall 1.000ms)",
        "[TurboQuant Profile] Layer 1 (Prefill) | TDOT: 11.000ms (host wall 1.000ms)",
        "[TurboQuant Profile] Layer 2 (Prefill) | TDOT: 11.000ms (host wall 1.000ms)",
    ]

    # The fourth layer is the next step's, and is still pending.
    collector.flush()
    assert len(records) == 2
    collector.flush()
    assert len(records) == 2  # nothing pending, nothing logged


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


@pytest.mark.parametrize(
    ("attributes", "expected"),
    [
        ({"layer_idx": 7}, "7"),
        ({"layer_name": "model.layers.11.self_attn.attn"}, "11"),
        ({"layer_name": "decoder.attn"}, "decoder.attn"),
        ({}, "?"),
    ],
)
def test_the_layer_label_is_an_index_wherever_one_can_be_found(attributes, expected):
    layer = type("Layer", (), attributes)()
    assert AscendTurboQuantDryRunBackendImpl._profile_layer_label(layer) == expected


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (AscendAttentionState.PrefillNoCache, "Prefill"),
        (AscendAttentionState.DecodeOnly, "Decode"),
        # Both of these take forward_impl's mixed route, and are named after it.
        (AscendAttentionState.ChunkedPrefill, "Mixed:ChunkedPrefill"),
        (AscendAttentionState.PrefillCacheHit, "Mixed:PrefillCacheHit"),
        (None, "Mixed"),
    ],
)
def test_the_phase_names_the_route_that_ran(state, expected):
    metadata = type("Metadata", (), {"attn_state": state})()
    assert AscendTurboQuantDryRunBackendImpl._profile_phase(metadata) == expected


# -- the plumbing around the profiler ----------------------------------------
#
# A fake torch_npu.profiler stands in for the real one, which needs a device.
# What is checked is the part that has to hold on hardware too: that the forward
# runs exactly once, that its summary reaches the collector, and that nothing
# the measurement does can change what the forward returns or raises.


class FakeProf:
    def __init__(self, rows):
        self.rows = rows
        self.entered = False

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *exc_info):
        return False

    def key_averages(self):
        if self.rows is None:
            raise RuntimeError("this torch_npu has no in-memory key_averages")
        return self.rows


@pytest.fixture
def profiled(monkeypatch):
    """An impl with profiling on, a fake profiler, and a captured collector."""
    import types

    import vllm_ascend.attention.turboquant_dry_run as module

    state: dict[str, object] = {"rows": ROWS, "calls": 0, "records": []}

    def profile(**kwargs):
        state["kwargs"] = kwargs
        state["prof"] = FakeProf(state["rows"])
        return state["prof"]

    monkeypatch.setattr(
        module,
        "torch_npu",
        types.SimpleNamespace(
            profiler=types.SimpleNamespace(
                ProfilerActivity=types.SimpleNamespace(CPU="cpu", NPU="npu"),
                profile=profile,
            )
        ),
    )

    collector = TurboQuantProfileLog()
    monkeypatch.setattr(module, "TURBOQUANT_PROFILE_LOG", collector)
    monkeypatch.setattr(module, "logger", types.SimpleNamespace(info=lambda *a: None, warning=lambda *a: None))

    impl = object.__new__(AscendTurboQuantDryRunBackendImpl)
    impl._profile = True

    def base_forward(self, **call):
        state["calls"] = int(state["calls"]) + 1
        if state.get("raises"):
            raise ValueError("the forward's own failure")
        return "attn_output"

    monkeypatch.setattr(module.AscendAttentionBackendImpl, "forward", base_forward, raising=False)
    state["impl"] = impl
    state["collector"] = collector
    state["module"] = module
    return state


def _run(profiled, **call):
    layer = type("Layer", (), {"layer_name": "model.layers.3.self_attn.attn"})()
    metadata = type("Metadata", (), {"attn_state": AscendAttentionState.DecodeOnly})()
    return profiled["impl"]._profiled_forward(layer, metadata, dict(call))


def test_a_profiled_forward_runs_once_and_records_one_summary(profiled):
    assert _run(profiled) == "attn_output"
    assert profiled["calls"] == 1
    assert profiled["prof"].entered
    assert profiled["kwargs"]["activities"] == ["cpu", "npu"]
    # No trace handler, and nothing that would write one.
    assert "on_trace_ready" not in profiled["kwargs"]

    pending = profiled["collector"]._pending
    assert list(pending) == ["3"]
    assert pending["3"].phase == "Decode"
    assert pending["3"].buckets[0].micros == pytest.approx(8000.0)
    assert pending["3"].host_wall_micros > 0.0


def test_a_profiler_that_cannot_be_read_switches_itself_off_and_keeps_the_output(profiled):
    profiled["rows"] = None  # key_averages() raises

    assert _run(profiled) == "attn_output"
    assert profiled["impl"]._profile is False
    assert profiled["collector"]._pending == {}


def test_an_unavailable_profiler_leaves_the_forward_alone(profiled, monkeypatch):
    import types

    # A torch_npu with no profiler at all: the forward still runs, unprofiled.
    monkeypatch.setattr(profiled["module"], "torch_npu", types.SimpleNamespace(profiler=None))

    assert _run(profiled) == "attn_output"
    assert profiled["calls"] == 1
    assert profiled["impl"]._profile is False
    assert profiled["collector"]._pending == {}


def test_the_forwards_own_failure_is_not_reported_as_a_profiler_failure(profiled):
    profiled["raises"] = True

    with pytest.raises(ValueError, match="the forward's own failure"):
        _run(profiled)
    # Still on: nothing about the profiler went wrong.
    assert profiled["impl"]._profile is True
