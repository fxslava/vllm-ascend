"""CAT 4 -- runtime faults & teardown: dummy kernel execution, deterministic
LIFO teardown, error recovery.

Covers the execution leg and every failure path around it: the surrogate
expert kernel's reference-checkable math, determinism and non-divergence
guarantees, operand-contract faults, zero-allocation execution and latency
emulation; engine-level execution reports; and the lifecycle manager's strict
LIFO teardown on normal exit, exceptions, ``atexit`` and signal traps.
"""

from __future__ import annotations

import signal
import time
from collections.abc import Callable
from contextlib import AbstractContextManager

import pytest
import torch

from ..benchmarks.capacity_probe import DummyLoopbackWeightProvider
from ..core.config import SANITY_GEOMETRY
from ..core.layout import ExpertTensorLayout
from ..core.slot_pool import StaticExpertSlotPool
from ..draft_inference.backends import MockV5Backend
from ..draft_inference.dry_run import DryRunConfig, parse_dry_run_config, run_draft_dry_run
from ..draft_inference.engine import DraftInferenceEngine
from ..hardware.dummy_kernel import (
    DEFAULT_EXPERT_LATENCY_US,
    DIGEST_MASK,
    INPUT_CLAMP_MAGNITUDE,
    MAX_REPRESENTATIVE_LATENCY_US,
    MIN_REPRESENTATIVE_LATENCY_US,
    DummyExpertKernelRunner,
)
from ..hardware.lifecycle import RuntimeLifecycleManager
from ..hardware.runtime import CpuRuntime
from ..protocols.provider import WeightProviderProtocol

# --------------------------------------------------------------------- helpers


class _ExpertOperands:
    """Deterministic slot-shaped weights/scales and scratchpad rows."""

    def __init__(self, layout: ExpertTensorLayout, config, seed: int):
        generator = torch.Generator().manual_seed(seed)
        self.weights = {
            name: torch.randint(0, 256, layout.spec_for(name).view_shape, dtype=torch.uint8, generator=generator)
            for name in ("w1", "w2", "w3")
        }
        self.scales = {
            name: torch.randint(
                0,
                256,
                (layout.spec_for(name).rows, layout.spec_for(name).cols // 32),
                dtype=torch.uint8,
                generator=generator,
            )
            for name in ("w1", "w2", "w3")
        }
        self.x = torch.randn(1, config.hidden_size, dtype=torch.bfloat16, generator=generator)
        self.gate_out = torch.empty(1, config.moe_intermediate_size, dtype=torch.bfloat16)
        self.up_out = torch.empty(1, config.moe_intermediate_size, dtype=torch.bfloat16)
        self.activated = torch.empty(1, config.moe_intermediate_size, dtype=torch.bfloat16)
        self.down_out = torch.empty(1, config.hidden_size, dtype=torch.bfloat16)

    def rows(self) -> dict[str, torch.Tensor]:
        return {
            "gate_out": self.gate_out,
            "up_out": self.up_out,
            "activated": self.activated,
            "down_out": self.down_out,
        }

    def run(self, kernel: DummyExpertKernelRunner, label: str = "cat4") -> None:
        kernel.execute_expert(
            self.x,
            self.weights["w1"],
            self.weights["w2"],
            self.weights["w3"],
            self.scales["w1"],
            self.scales["w2"],
            self.scales["w3"],
            self.gate_out,
            self.up_out,
            self.activated,
            self.down_out,
            label,
        )


def _reference_outputs(ops: _ExpertOperands):
    """Recompute the surrogate math op-for-op (same dtypes, same order)."""

    def digest(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        row = weight.sum(dim=1, dtype=torch.int32) + scale.sum(dim=1, dtype=torch.int32)
        return row.bitwise_and(DIGEST_MASK)

    x_sanitized = torch.nan_to_num(ops.x, nan=0.0, posinf=0.0, neginf=0.0).clamp(
        -INPUT_CLAMP_MAGNITUDE, INPUT_CLAMP_MAGNITUDE
    )
    x_factor = torch.sum(x_sanitized, dim=1, dtype=torch.bfloat16).abs()
    gate = digest(ops.weights["w1"], ops.scales["w1"]).to(torch.bfloat16) * x_factor
    up = digest(ops.weights["w3"], ops.scales["w3"]).to(torch.bfloat16) * x_factor
    activated = torch.sigmoid(gate)
    activated = activated * gate
    activated = activated * up
    down_factor = torch.sum(activated.view(1, -1), dim=1, dtype=torch.bfloat16)
    down = digest(ops.weights["w2"], ops.scales["w2"]).to(torch.bfloat16) * down_factor
    return {
        "gate_out": gate.view(1, -1),
        "up_out": up.view(1, -1),
        "activated": activated.view(1, -1),
        "down_out": down.view(1, -1),
    }


def _build_engine(provider: WeightProviderProtocol, sanity_config, sanity_layout: ExpertTensorLayout):
    return DraftInferenceEngine(
        config=sanity_config,
        layout=sanity_layout,
        provider=provider,
        backend=MockV5Backend(),
        num_slots=32,
        device="cpu",
    )


# --------------------------------------------------------- backend contracts


def test_mock_backend_quant_gemm_contract(sanity_layout: ExpertTensorLayout) -> None:
    backend = MockV5Backend()
    spec = sanity_layout.spec_for("w1")
    x = torch.empty(1, 128, dtype=torch.bfloat16)
    out = torch.empty(1, spec.rows, dtype=torch.bfloat16)
    weight = torch.zeros(spec.view_shape, dtype=torch.uint8)
    scale = torch.zeros((spec.rows, spec.cols // 32), dtype=torch.uint8)

    backend.quant_gemm(x, weight, scale, out, label="w1")
    assert torch.all(out != 0)  # deterministic pattern was written in place
    with pytest.raises(ValueError, match="packed weight"):
        backend.quant_gemm(torch.empty(1, 64, dtype=torch.bfloat16), weight, scale, out, label="w1-bad-k")
    with pytest.raises(ValueError, match="scale"):
        backend.quant_gemm(x, weight, torch.zeros(4, 2, dtype=torch.uint8), out, label="w1-bad-scale")


# --------------------------------------------------- surrogate kernel: happy


def test_dummy_kernel_matches_reference_math(sanity_config, sanity_layout: ExpertTensorLayout) -> None:
    ops = _ExpertOperands(sanity_layout, sanity_config, seed=42)
    kernel = DummyExpertKernelRunner(sanity_layout)
    ops.run(kernel)
    expected = _reference_outputs(ops)
    for name, reference in expected.items():
        written = ops.rows()[name]
        assert torch.equal(written, reference), f"{name}: surrogate diverged from its reference math"
        assert bool(torch.isfinite(written).all()), name
    assert kernel.executions == 1


def test_dummy_kernel_deterministic_and_non_divergent(sanity_config, sanity_layout: ExpertTensorLayout) -> None:
    kernel = DummyExpertKernelRunner(sanity_layout)
    ops = _ExpertOperands(sanity_layout, sanity_config, seed=7)
    other = _ExpertOperands(sanity_layout, sanity_config, seed=8)

    ops.run(kernel)
    first = {name: tensor.clone() for name, tensor in ops.rows().items()}
    ops.run(kernel)
    second = {name: tensor.clone() for name, tensor in ops.rows().items()}
    assert all(torch.equal(first[name], second[name]) for name in first)  # bit-stable reruns

    # Different slot bytes must yield different outputs (expert-distinguishing).
    kernel.execute_expert(
        other.x,
        other.weights["w1"],
        other.weights["w2"],
        other.weights["w3"],
        other.scales["w1"],
        other.scales["w2"],
        other.scales["w3"],
        other.gate_out,
        other.up_out,
        other.activated,
        other.down_out,
        "other",
    )
    assert any(not torch.equal(first[name], other.rows()[name]) for name in first)

    # Non-divergent even on poisoned input activations: NaN/Inf never propagate.
    poisoned = torch.full((1, sanity_config.hidden_size), float("nan"), dtype=torch.bfloat16)
    poisoned[0, 0] = float("inf")
    poisoned[0, 1] = 1e38
    kernel.execute_expert(
        poisoned,
        ops.weights["w1"],
        ops.weights["w2"],
        ops.weights["w3"],
        ops.scales["w1"],
        ops.scales["w2"],
        ops.scales["w3"],
        ops.gate_out,
        ops.up_out,
        ops.activated,
        ops.down_out,
        "poison",
    )
    assert all(bool(torch.isfinite(tensor).all()) for tensor in ops.rows().values())


def test_dummy_kernel_zero_allocation(
    sanity_config, sanity_layout: ExpertTensorLayout, allocation_guard: Callable[..., AbstractContextManager[None]]
) -> None:
    kernel = DummyExpertKernelRunner(sanity_layout)
    ops = _ExpertOperands(sanity_layout, sanity_config, seed=9)
    ops.run(kernel)  # warm up lazily-initialized torch state first
    with allocation_guard():
        for _ in range(5):
            ops.run(kernel)
    assert kernel.executions == 6


def test_dummy_kernel_latency_band(sanity_config, sanity_layout: ExpertTensorLayout) -> None:
    assert MIN_REPRESENTATIVE_LATENCY_US <= DEFAULT_EXPERT_LATENCY_US <= MAX_REPRESENTATIVE_LATENCY_US
    kernel = DummyExpertKernelRunner(sanity_layout, latency_us=MIN_REPRESENTATIVE_LATENCY_US)
    ops = _ExpertOperands(sanity_layout, sanity_config, seed=11)

    start = time.perf_counter()
    for _ in range(3):
        ops.run(kernel)
    elapsed_us = (time.perf_counter() - start) * 1e6

    budget_us = 3 * MIN_REPRESENTATIVE_LATENCY_US
    assert elapsed_us >= budget_us, "latency emulation burned less than the requested budget"
    assert elapsed_us < budget_us + 15_000.0, f"latency emulation overshot: {elapsed_us:.0f}us"
    assert kernel.burned_us_total == pytest.approx(budget_us)
    assert DummyExpertKernelRunner(sanity_layout, latency_us=0.0).latency_us == 0.0
    with pytest.raises(ValueError, match="latency_us"):
        DummyExpertKernelRunner(sanity_layout, latency_us=-5.0)


# ---------------------------------------------------- surrogate kernel: faults


def test_dummy_kernel_rejects_malformed_operands(sanity_config, sanity_layout: ExpertTensorLayout) -> None:
    kernel = DummyExpertKernelRunner(sanity_layout)
    ops = _ExpertOperands(sanity_layout, sanity_config, seed=13)

    bad_shape_x = torch.empty(1, sanity_config.hidden_size + 2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match=r"slot-bad/x: expected shape"):
        kernel.execute_expert(
            bad_shape_x,
            ops.weights["w1"],
            ops.weights["w2"],
            ops.weights["w3"],
            ops.scales["w1"],
            ops.scales["w2"],
            ops.scales["w3"],
            ops.gate_out,
            ops.up_out,
            ops.activated,
            ops.down_out,
            "slot-bad",
        )

    wrong_dtype_gate = torch.empty(1, sanity_config.moe_intermediate_size, dtype=torch.float32)
    with pytest.raises(ValueError, match=r"slot-bad/gate_out: expected dtype"):
        kernel.execute_expert(
            ops.x,
            ops.weights["w1"],
            ops.weights["w2"],
            ops.weights["w3"],
            ops.scales["w1"],
            ops.scales["w2"],
            ops.scales["w3"],
            wrong_dtype_gate,
            ops.up_out,
            ops.activated,
            ops.down_out,
            "slot-bad",
        )

    # Transposed then row-sliced: [1, I] whose last dim has stride 2.
    non_contiguous_up = torch.empty(sanity_config.moe_intermediate_size, 2, dtype=torch.bfloat16).t()[0:1]
    with pytest.raises(ValueError, match=r"slot-bad/up_out: operand must be contiguous"):
        kernel.execute_expert(
            ops.x,
            ops.weights["w1"],
            ops.weights["w2"],
            ops.weights["w3"],
            ops.scales["w1"],
            ops.scales["w2"],
            ops.scales["w3"],
            ops.gate_out,
            non_contiguous_up,
            ops.activated,
            ops.down_out,
            "slot-bad",
        )

    bad_scale = torch.zeros((3, 3), dtype=torch.uint8)
    with pytest.raises(ValueError, match=r"slot-bad/w2_scale: expected shape"):
        kernel.execute_expert(
            ops.x,
            ops.weights["w1"],
            ops.weights["w2"],
            ops.weights["w3"],
            ops.scales["w1"],
            bad_scale,
            ops.scales["w3"],
            ops.gate_out,
            ops.up_out,
            ops.activated,
            ops.down_out,
            "slot-bad",
        )
    # Nothing was written: fault happens before any byte moves.
    assert kernel.executions == 0


# --------------------------------------------------------- engine dry run leg


def test_draft_engine_single_step_shapes(sanity_config, sanity_layout: ExpertTensorLayout) -> None:
    provider = DummyLoopbackWeightProvider(CpuRuntime(), sanity_layout)
    engine = _build_engine(provider, sanity_config, sanity_layout)

    engine.prepare_step(token_id=5)
    report = engine.step()

    assert report.layers_executed == sanity_config.num_layers
    assert report.layer_hits + report.layer_misses == sanity_config.num_layers * sanity_config.top_k
    # Early layers are churned out of the shared 32-slot pool by the score
    # layers that follow; the route buffer now holds the FINAL layer's top-k,
    # and those experts must still be resident with the loopback pattern.
    last_layer = sanity_config.num_layers - 1
    last_experts = [int(value) for value in engine._scratchpad["route_experts"][0].tolist()]
    for expert_id in last_experts:
        assert engine.pool.slot_of(last_layer, expert_id) >= 0
    slot = engine.pool.slot_of(last_layer, last_experts[0])
    assert int(engine.pool.weight_views(slot)["w1"][0, 0]) == 0xA7
    assert engine.arena_fingerprint() == engine.initial_fingerprint
    # The surrogate kernel populated every scratchpad row it owns, finitely.
    for name in ("gate_out", "up_out", "activated", "down_out", "routed_accum", "hidden"):
        tensor = engine._scratchpad[name]
        assert bool(torch.isfinite(tensor).all()), name
    assert bool((engine._scratchpad["routed_accum"] != 0).any())
    assert engine.expert_kernel.executions == sanity_config.num_layers * sanity_config.top_k


def test_draft_engine_hits_and_misses_progression(sanity_config, sanity_layout: ExpertTensorLayout) -> None:
    provider = DummyLoopbackWeightProvider(CpuRuntime(), sanity_layout)
    engine = _build_engine(provider, sanity_config, sanity_layout)
    engine.prepare_step(token_id=1)
    first = engine.step()
    second = engine.step()
    # Both runs execute the full layer stack over the same hash-routed token.
    assert first.layers_executed == second.layers_executed == sanity_config.num_layers
    assert first.layer_hits + first.layer_misses == second.layer_hits + second.layer_misses
    # Error recovery: a rejected step leaves the engine fully usable.
    with pytest.raises(ValueError):
        engine.prepare_step(token_id=-1)
    engine.prepare_step(token_id=3)
    assert engine.step().layers_executed == sanity_config.num_layers


def test_dry_run_config_validation() -> None:
    with pytest.raises(ValueError):
        parse_dry_run_config(["--steps", "0"])
    with pytest.raises(ValueError):
        parse_dry_run_config(["--pool-slots", "2"])
    with pytest.raises(ValueError):
        parse_dry_run_config(["--expert-latency-us", "-1"])
    config = parse_dry_run_config(["--device", "cpu", "--steps", "3", "--small-geometry"])
    assert config.device == "cpu" and config.steps == 3 and config.small_geometry
    assert config.expert_latency_us == 0.0


def test_dry_run_cpu_end_to_end(sanity_config, sanity_layout: ExpertTensorLayout, mock_provider_factory) -> None:
    provider = mock_provider_factory(
        sanity_layout, layer_ids=tuple(range(sanity_config.num_layers)), num_experts=sanity_config.num_routed_experts
    )
    report = run_draft_dry_run(
        DryRunConfig(device="cpu", steps=10, pool_slots=32, attention_heads=8, small_geometry=True, report_json=None),
        provider,
    )
    assert report["verdict"] == "PASS"
    assert report["zero_allocation"]["violations"] == []
    assert report["zero_allocation"]["fingerprints_stable"] is True
    assert report["moe_gemm_backend"] == "dummy-surrogate-kernel"
    assert report["expert_kernel"]["executions"] == (10 + 1) * sanity_config.num_layers * sanity_config.top_k
    teardown = report["teardown"]
    assert teardown["ok"] is True and teardown["lifo_ok"] is True and teardown["errors"] == []
    assert teardown["steps"][0] == "slot-pool-arena"  # registered last, torn down first


def test_cpu_runtime_loopback_provider_composes(sanity_config, sanity_layout: ExpertTensorLayout) -> None:
    """The loopback DMA provider doubles as a draft provider on any host."""
    provider = DummyLoopbackWeightProvider(CpuRuntime(), sanity_layout)
    pool = StaticExpertSlotPool(SANITY_GEOMETRY, num_slots=8, layout=sanity_layout)
    reservation = pool.acquire_for_step(0, [0, 1, 2, 3, 4, 5], provider)
    assert len(reservation.slot_ids) == 6
    for param_key, view in pool.param_views(reservation.slot_ids[0]).items():
        assert torch.equal(view, provider.pinned_cpu_weight(0, 0, param_key))
    pool.release_step(reservation)


# ------------------------------------------------------ lifecycle teardown leg


def test_lifecycle_strict_lifo_teardown_and_idempotency() -> None:
    manager = RuntimeLifecycleManager(CpuRuntime(), trap_signals=False)
    order: list[str] = []
    manager.push("stream-a", lambda: order.append("stream-a"), kind="stream")
    manager.push("workspace-b", lambda: order.append("workspace-b"), kind="workspace")
    manager.push("arena-c", lambda: order.append("arena-c"), kind="arena")
    assert manager.active_resources == ["stream-a", "workspace-b", "arena-c"]

    report = manager.teardown(trigger="manual")
    assert order == ["arena-c", "workspace-b", "stream-a"]  # strict LIFO
    assert report.lifo_order_respected and report.ok
    assert report.steps[-1].kind == "device"  # device epilogue runs last
    assert [step.name for step in report.steps if not step.kind.startswith("device")] == order

    # Idempotent: the same report comes back; the stack is gone.
    assert manager.teardown(trigger="again") is report
    assert manager.active_resources == []
    with pytest.raises(RuntimeError, match="already torn down"):
        manager.push("late", lambda: None)

    duplicate = RuntimeLifecycleManager(CpuRuntime(), trap_signals=False)
    duplicate.push("same-name", lambda: None)
    with pytest.raises(ValueError, match="named"):
        duplicate.push("same-name", lambda: None)


def test_lifecycle_teardown_survives_step_failure() -> None:
    manager = RuntimeLifecycleManager(CpuRuntime(), trap_signals=False)
    order: list[str] = []

    def exploding() -> None:
        order.append("boom")
        raise RuntimeError("synthetic teardown failure")

    manager.push("first", lambda: order.append("first"))
    manager.push("exploding", exploding)
    manager.push("last", lambda: order.append("last"))

    report = manager.teardown()
    assert order == ["last", "boom", "first"]  # one failure never aborts the chain
    assert not report.ok
    assert report.lifo_order_respected  # order was still strictly LIFO
    assert report.errors == ["exploding: RuntimeError: synthetic teardown failure"]


def test_lifecycle_session_teardown_on_exception() -> None:
    manager = RuntimeLifecycleManager(CpuRuntime(), trap_signals=False)
    seen: list[str] = []
    manager.push("resource", lambda: seen.append("resource"))

    with pytest.raises(ValueError, match="mid-run fault"), manager.session():
        manager.push("inner", lambda: seen.append("inner"))
        raise ValueError("mid-run fault")

    assert seen == ["inner", "resource"]  # LIFO even on the exception path
    assert manager.last_report is not None
    assert manager.last_report.trigger == "exception:ValueError"
    assert manager.last_report.ok


def test_lifecycle_atexit_hook_teardown() -> None:
    manager = RuntimeLifecycleManager(CpuRuntime(), trap_signals=False)
    released: list[str] = []
    manager.push("dma-stream", lambda: released.append("dma-stream"), kind="stream")
    manager.register_workspace("aclnn-workspace", release=lambda: released.append("aclnn-workspace"))
    manager.register_arena("slot-arena", release=lambda: released.append("slot-arena"))

    manager._atexit_teardown()  # what atexit would invoke at interpreter exit
    assert released == ["slot-arena", "aclnn-workspace", "dma-stream"]
    assert manager.last_report is not None and manager.last_report.trigger == "atexit"
    # A second atexit pass (e.g. duplicate registration) is a no-op.
    manager._atexit_teardown()
    assert released.count("slot-arena") == 1


def test_lifecycle_device_epilogue_runs_through_runtime_seam() -> None:
    calls: list[str] = []

    class RecordingRuntime(CpuRuntime):
        def synchronize_device(self) -> None:
            calls.append("sync")

        def release_cache(self) -> None:
            calls.append("cache")

        def reset_device(self) -> None:
            calls.append("reset")

    manager = RuntimeLifecycleManager(RecordingRuntime(), trap_signals=False)
    manager.push("solo", lambda: None)
    report = manager.teardown()
    assert calls == ["sync", "cache", "reset"]
    assert [step.name for step in report.steps] == [
        "solo",
        "device/synchronize",
        "device/release-cache",
        "device/reset",
    ]
    assert report.ok


def test_lifecycle_signal_traps_teardown_then_default() -> None:
    saved_int = signal.getsignal(signal.SIGINT)
    saved_term = signal.getsignal(signal.SIGTERM)
    manager = RuntimeLifecycleManager(CpuRuntime(), trap_signals=True)
    torn: list[str] = []
    manager.push("signal-owned", lambda: torn.append("signal-owned"))

    try:
        assert callable(signal.getsignal(signal.SIGINT))
        assert callable(signal.getsignal(signal.SIGTERM))

        with pytest.raises(KeyboardInterrupt):
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert torn == ["signal-owned"]
        assert manager.last_report is not None
        assert manager.last_report.trigger == f"signal:{signal.SIGINT}"

        manager2 = RuntimeLifecycleManager(CpuRuntime(), trap_signals=True)
        torn2: list[str] = []
        manager2.push("term-owned", lambda: torn2.append("term-owned"))
        with pytest.raises(SystemExit) as excinfo:
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert excinfo.value.code == 128 + signal.SIGTERM
        assert torn2 == ["term-owned"]
        assert manager2.last_report is not None
        assert manager2.last_report.trigger == f"signal:{signal.SIGTERM}"
    finally:
        manager.uninstall_signal_traps()
        assert signal.getsignal(signal.SIGINT) == saved_int
        assert signal.getsignal(signal.SIGTERM) == saved_term
