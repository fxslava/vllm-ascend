"""Validation for the NPU offload stress harness (CPU-only, synthetic tensors).

Covers the pieces the pilot suite does not: argument parsing/presets, the
mixture-Zipf trace model (hash vs score layers, hot-head share), the
DMA-capable pinned host storage, transactional streamed fills into the slot
pool, and the full CPU dry run end-to-end.
"""

from __future__ import annotations

import json

import bench_npu_offload_stress as bench
import pytest
import torch
from bench_npu_offload_stress import (
    AscendPinnedHostStorage,
    CpuRuntime,
    OffloadStressHarness,
    RouterTraceSimulator,
    parse_config,
)
from slot_pool import ExpertTensorLayout, SlotFillProviderProtocol, StaticExpertSlotPool

SANITY_CONFIG = bench.SANITY_GEOMETRY
SANITY_LAYOUT = ExpertTensorLayout.for_deepseek_v4_flash(SANITY_CONFIG)


# --------------------------------------------------------------------------- #
# Argument parsing & presets                                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "argv",
    [
        ["--steps", "0"],
        ["--pool-slots", "4"],
        ["--pinned-layers", "50"],
        ["--pinned-experts-per-layer", "0"],
        ["--hot-ratio", "1.5"],
        ["--zipf-exponent", "-1"],
        ["--hot-experts", "256"],
        ["--buckets", "0"],
    ],
)
def test_argument_parsing_rejects_invalid(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        parse_config(argv)
    assert excinfo.value.code == 2


def test_argument_parsing_presets_and_overrides() -> None:
    npu_config = parse_config([])
    assert npu_config.device == "npu:0"
    assert npu_config.steps == 2000 and npu_config.pool_slots == 3200
    assert npu_config.small_geometry is False and npu_config.pinned_layers == 43

    dry_config = parse_config(["--dry-run"])
    assert dry_config.device == "cpu"
    assert dry_config.steps == 50 and dry_config.pool_slots == 16
    assert dry_config.small_geometry is True and dry_config.pinned_layers == 2
    assert dry_config.verify_samples == 8

    overridden = parse_config(["--dry-run", "--steps", "10", "--seed", "3"])
    assert overridden.steps == 10 and overridden.seed == 3
    assert overridden.device == "cpu"  # preset still applies to unspecified flags


# --------------------------------------------------------------------------- #
# Router trace model                                                           #
# --------------------------------------------------------------------------- #


def test_trace_simulator_layers_hash_split_and_determinism() -> None:
    simulator = RouterTraceSimulator(
        SANITY_CONFIG, num_experts_available=256, hot_experts=64, hot_ratio=0.35, zipf_exponent=1.2, seed=7
    )
    tokens = 40
    trace = simulator.generate_trace(tokens * SANITY_CONFIG.num_layers)

    assert {step.layer_idx for step in trace} == set(range(SANITY_CONFIG.num_layers))
    hash_steps = [step for step in trace if step.routed_by == "hash"]
    score_steps = [step for step in trace if step.routed_by == "score"]
    assert len(hash_steps) == SANITY_CONFIG.num_hash_layers * tokens
    assert len(score_steps) == (SANITY_CONFIG.num_layers - SANITY_CONFIG.num_hash_layers) * tokens
    for step in trace:
        assert len(step.expert_ids) == SANITY_CONFIG.top_k
        assert len(set(step.expert_ids)) == SANITY_CONFIG.top_k
        assert all(0 <= expert < 256 for expert in step.expert_ids)

    twin = RouterTraceSimulator(
        SANITY_CONFIG, num_experts_available=256, hot_experts=64, hot_ratio=0.35, zipf_exponent=1.2, seed=7
    )
    assert twin.generate_trace(len(trace)) == trace  # same seed: identical trace
    other = RouterTraceSimulator(
        SANITY_CONFIG, num_experts_available=256, hot_experts=64, hot_ratio=0.35, zipf_exponent=1.2, seed=8
    )
    assert other.generate_trace(len(trace)) != trace  # different seed: different trace


def test_trace_simulator_hot_head_share() -> None:
    simulator = RouterTraceSimulator(
        SANITY_CONFIG, num_experts_available=256, hot_experts=64, hot_ratio=0.35, zipf_exponent=1.2, seed=11
    )
    trace = simulator.generate_trace(43 * 200)
    hot = set(simulator.hot_expert_ids)
    draws = [expert for step in trace for expert in step.expert_ids]
    hot_share = sum(1 for expert in draws if expert in hot) / len(draws)
    assert 0.25 <= hot_share <= 0.45, f"hot share {hot_share:.3f} outside the mixture band"


def test_trace_simulator_validates_geometry() -> None:
    with pytest.raises(ValueError):
        RouterTraceSimulator(
            SANITY_CONFIG, num_experts_available=256, hot_experts=256, hot_ratio=0.35, zipf_exponent=1.0, seed=1
        )  # empty cold tail
    with pytest.raises(ValueError):
        RouterTraceSimulator(
            SANITY_CONFIG, num_experts_available=256, hot_experts=8, hot_ratio=5.0, zipf_exponent=1.0, seed=1
        )
    with pytest.raises(ValueError):
        RouterTraceSimulator(
            SANITY_CONFIG,
            num_experts_available=256,
            hot_experts=8,
            hot_ratio=0.35,
            zipf_exponent=1.0,
            seed=1,
            num_layers=2,
            num_hash_layers=3,
        )


# --------------------------------------------------------------------------- #
# Pinned host storage & streamed DMA fills                                     #
# --------------------------------------------------------------------------- #


def test_pinned_storage_stages_and_fills_views() -> None:
    runtime = CpuRuntime()
    storage = AscendPinnedHostStorage(runtime, SANITY_LAYOUT, layer_ids=[0, 1], experts_per_layer=8)
    assert isinstance(storage, SlotFillProviderProtocol)
    assert storage.staged_expert_count == 16
    assert storage.staged_bytes() == 16 * SANITY_LAYOUT.slot_num_bytes

    storage.ensure_staged(1, 7)
    with pytest.raises(KeyError):
        storage.ensure_staged(2, 0)  # layer 2 not staged
    with pytest.raises(KeyError):
        storage.ensure_staged(1, 8)  # expert id beyond staged range

    views = {spec.param_key: torch.zeros(spec.view_shape, dtype=torch.uint8) for spec in SANITY_LAYOUT.specs}
    moved = storage.fill_slot_params(1, 7, views)
    assert moved == SANITY_LAYOUT.slot_num_bytes
    assert storage.bytes_copied == SANITY_LAYOUT.slot_num_bytes
    for param_key, view in views.items():
        assert torch.equal(view, storage.pinned_cpu_weight(1, 7, param_key))


def test_streamed_fill_is_transactional_on_unstaged_expert() -> None:
    pool = StaticExpertSlotPool(SANITY_CONFIG, num_slots=6)
    storage = AscendPinnedHostStorage(CpuRuntime(), SANITY_LAYOUT, layer_ids=[0], experts_per_layer=8)

    first = pool.acquire_for_step(0, [0, 1, 2, 3, 4, 5], storage)
    assert pool.stats.loads == 6
    assert pool.stats.bytes_staged == 6 * SANITY_LAYOUT.slot_num_bytes
    resident = {expert: pool.slot_of(0, expert) for expert in range(6)}
    for expert, slot in resident.items():
        for param_key, view in pool.param_views(slot).items():
            assert torch.equal(view, storage.pinned_cpu_weight(0, expert, param_key))
    pool.release_step(first)

    # Expert ids 100+ are not staged: ensure_staged must fail BEFORE the ledger
    # commits, so residency, tables and counters are untouched.
    with pytest.raises(KeyError):
        pool.acquire_for_step(0, [100, 101, 102, 103, 104, 105], storage)
    assert {expert: pool.slot_of(0, expert) for expert in range(6)} == resident
    assert pool.stats.evictions == 0 and pool.stats.loads == 6
    assert pool.stats.bytes_staged == 6 * SANITY_LAYOUT.slot_num_bytes


# --------------------------------------------------------------------------- #
# End-to-end CPU dry run                                                       #
# --------------------------------------------------------------------------- #


def test_dry_run_end_to_end(tmp_path) -> None:
    report_path = tmp_path / "dry_run_report.json"
    exit_code = bench.main(["--dry-run", f"--report-json={report_path}", "--seed", "7"])
    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["verdict"] == "PASS"
    assert report["zero_allocation"]["fingerprints_stable"] is True
    assert report["zero_allocation"]["allocator_invariant_ok"] is None  # cpu: no device allocator
    assert report["zero_allocation"]["verification"]["ok"] is True
    assert len(report["warmup_curve"]) == 5
    assert report["bandwidth_gbps"]["aggregate_gbps"] > 0
    assert report["config"]["steps"] == 50


def test_harness_small_run_matches_report() -> None:
    config = parse_config(["--dry-run", "--steps", "20", "--verify-samples", "3", "--buckets", "4"])
    report = OffloadStressHarness(config).run()
    assert len(report.metrics) == 20
    assert report.invariants_ok()
    assert report.verification is not None and report.verification[0]
    expected_hit_rate = sum(SANITY_CONFIG.top_k - metric.loads for metric in report.metrics) / (
        20 * SANITY_CONFIG.top_k
    )
    assert report.hit_rate_overall == pytest.approx(expected_hit_rate)
