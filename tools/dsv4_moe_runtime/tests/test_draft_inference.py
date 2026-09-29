"""Draft inference package tests: V5 wrapper contracts and the engine dry run."""

from __future__ import annotations

import pytest
import torch

from dsv4_moe_runtime.benchmarks.capacity_probe import DummyLoopbackWeightProvider
from dsv4_moe_runtime.core.layout import ExpertTensorLayout
from dsv4_moe_runtime.core.slot_pool import StaticExpertSlotPool
from dsv4_moe_runtime.draft_inference.backends import MockV5Backend
from dsv4_moe_runtime.draft_inference.dry_run import parse_dry_run_config, run_draft_dry_run
from dsv4_moe_runtime.draft_inference.engine import DraftInferenceEngine
from dsv4_moe_runtime.hardware.runtime import CpuRuntime
from dsv4_moe_runtime.protocols.provider import WeightProviderProtocol


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


def _build_engine(provider: WeightProviderProtocol, sanity_config, sanity_layout: ExpertTensorLayout):
    return DraftInferenceEngine(
        config=sanity_config,
        layout=sanity_layout,
        provider=provider,
        backend=MockV5Backend(),
        num_slots=32,
        device="cpu",
    )


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


def test_draft_engine_zero_allocation_50_steps(
    sanity_config, sanity_layout: ExpertTensorLayout, mock_provider_factory, allocation_guard
) -> None:
    provider = mock_provider_factory(
        sanity_layout, layer_ids=tuple(range(sanity_config.num_layers)), num_experts=sanity_config.num_routed_experts
    )
    engine = _build_engine(provider, sanity_config, sanity_layout)
    engine.warm_up_routing_tables()
    fingerprint = engine.arena_fingerprint()

    with allocation_guard():
        for step_index in range(50):
            engine.prepare_step(token_id=(step_index * 13) % sanity_config.vocab_size)
            report = engine.step()
            assert report.layers_executed == sanity_config.num_layers
            assert engine.arena_fingerprint() == fingerprint, f"step {step_index}: an arena moved"

    assert report.layer_misses >= 0
    assert engine.arena_fingerprint() == fingerprint


def test_draft_engine_hits_and_misses_progression(sanity_config, sanity_layout: ExpertTensorLayout) -> None:
    provider = DummyLoopbackWeightProvider(CpuRuntime(), sanity_layout)
    engine = _build_engine(provider, sanity_config, sanity_layout)
    engine.prepare_step(token_id=1)
    first = engine.step()
    second = engine.step()
    # Both runs execute the full layer stack over the same hash-routed token.
    assert first.layers_executed == second.layers_executed == sanity_config.num_layers
    assert first.layer_hits + first.layer_misses == second.layer_hits + second.layer_misses
    with pytest.raises(ValueError):
        engine.prepare_step(token_id=-1)


def test_dry_run_config_validation() -> None:
    with pytest.raises(ValueError):
        parse_dry_run_config(["--steps", "0"])
    with pytest.raises(ValueError):
        parse_dry_run_config(["--pool-slots", "2"])
    config = parse_dry_run_config(["--device", "cpu", "--steps", "3", "--small-geometry"])
    assert config.device == "cpu" and config.steps == 3 and config.small_geometry


def test_dry_run_cpu_end_to_end(sanity_config, sanity_layout: ExpertTensorLayout, mock_provider_factory) -> None:
    provider = mock_provider_factory(
        sanity_layout, layer_ids=tuple(range(sanity_config.num_layers)), num_experts=sanity_config.num_routed_experts
    )
    from dsv4_moe_runtime.draft_inference.dry_run import DryRunConfig

    report = run_draft_dry_run(
        DryRunConfig(device="cpu", steps=10, pool_slots=32, attention_heads=8, small_geometry=True, report_json=None),
        provider,
    )
    assert report["verdict"] == "PASS"
    assert report["zero_allocation"]["violations"] == []
    assert report["zero_allocation"]["fingerprints_stable"] is True
    assert report["moe_gemm_backend"] == "mock"


def test_cpu_runtime_loopback_provider_composes(sanity_layout: ExpertTensorLayout) -> None:
    """The loopback DMA provider doubles as a draft provider on any host."""
    provider = DummyLoopbackWeightProvider(CpuRuntime(), sanity_layout)
    pool = StaticExpertSlotPool(sanity_config_provider(), num_slots=8, layout=sanity_layout)
    reservation = pool.acquire_for_step(0, [0, 1, 2, 3, 4, 5], provider)
    assert len(reservation.slot_ids) == 6
    for param_key, view in pool.param_views(reservation.slot_ids[0]).items():
        assert torch.equal(view, provider.pinned_cpu_weight(0, 0, param_key))
    pool.release_step(reservation)


def sanity_config_provider():
    from dsv4_moe_runtime.core.config import SANITY_GEOMETRY

    return SANITY_GEOMETRY
