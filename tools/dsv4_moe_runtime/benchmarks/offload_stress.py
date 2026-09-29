"""Offload stress harness: config, synthetic-run orchestration and CLI main.

Orchestrates one end-to-end measurement over the memory subsystem only:
acquire -> streamed DMA fill -> stream sync -> release, with a per-step
hardware zero-allocation check. No GEMMs, no attention, no LLM forward.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..benchmarks.stress_config import BenchConfig, parse_config, print_plan  # re-exported
from ..benchmarks.telemetry import (
    StepMetric,
    StressReport,
    base_fingerprints,
    full_fingerprints,
    render_report,
    report_to_dict,
)
from ..benchmarks.trace_simulator import RouterTraceSimulator, TraceStep
from ..core.slot_pool import UNRESIDENT_SLOT_ID, StaticExpertSlotPool
from ..hardware.pinned_storage import AscendPinnedHostStorage
from ..hardware.runtime import DeviceRuntime, make_runtime


class OffloadStressHarness:
    """Orchestrates one end-to-end synthetic offload measurement."""

    def __init__(self, config: BenchConfig):
        self._config = config

    def run(self) -> StressReport:
        started = time.perf_counter()
        config = self._config
        model_config = config.model_config
        layout = config.layout
        traced_layers, addressable_experts, hash_layers = config.trace_plan()

        runtime = make_runtime(config.device)
        storage = AscendPinnedHostStorage(
            runtime,
            layout,
            layer_ids=range(traced_layers),
            experts_per_layer=config.pinned_experts_per_layer,
            pin=config.pin_host_memory,
        )
        pool = StaticExpertSlotPool(model_config, config.pool_slots, layout=layout, device=runtime.device)
        simulator = RouterTraceSimulator(
            model_config,
            num_experts_available=addressable_experts,
            hot_experts=config.hot_experts,
            hot_ratio=config.hot_ratio,
            zipf_exponent=config.zipf_exponent,
            seed=config.seed,
            num_layers=traced_layers,
            num_hash_layers=hash_layers,
        )
        trace = simulator.generate_trace(config.steps)
        print(
            f"run: staged {storage.staged_expert_count} experts "
            f"({storage.staged_bytes() / 2**30:.2f} GiB, {'pinned' if storage.pinned else 'pageable'}), "
            f"pool {pool.num_slots} slots, trace {len(trace)} steps generated",
            flush=True,
        )

        runtime.synchronize_device()
        loop = _execute_trace_loop(pool, storage, runtime, trace, config.steps)

        verification = None
        if config.verify_samples:
            verification = _verify_sampled_slots(pool, storage, trace, config.verify_samples, config.seed)

        return StressReport(
            config=config,
            slot_num_bytes=layout.slot_num_bytes,
            experts_staged=storage.staged_expert_count,
            host_staged_bytes=storage.staged_bytes(),
            pool_bytes=config.pool_slots * layout.slot_num_bytes,
            steps=config.steps,
            top_k=model_config.top_k,
            baseline_allocated=loop.baseline_allocated,
            baseline_reserved=loop.baseline_reserved,
            final_allocated=loop.final_allocated,
            final_reserved=loop.final_reserved,
            allocation_violation_steps=loop.allocation_violation_steps,
            fingerprints_stable=loop.fingerprints_stable,
            verification=verification,
            metrics=loop.metrics,
            wall_time_s=time.perf_counter() - started,
        )


@dataclass
class _TraceRunResult:
    """Raw outcome of the measured trace loop (pre-report aggregation)."""

    metrics: list[StepMetric]
    allocation_violation_steps: list[int]
    fingerprints_stable: bool
    baseline_allocated: int | None
    baseline_reserved: int | None
    final_allocated: int | None
    final_reserved: int | None


def _execute_trace_loop(
    pool: StaticExpertSlotPool,
    storage: AscendPinnedHostStorage,
    runtime: DeviceRuntime,
    trace: Sequence[TraceStep],
    total_steps: int,
) -> _TraceRunResult:
    """Measured section: acquire -> DMA sync -> release, with invariant checks.

    Zero-allocation is enforced per step via the runtime allocator (when the
    hardware provides accounting) and via data_ptr fingerprint sweeps (light
    sweep per step, full view sweep every ``total_steps // 20`` steps).
    """
    allocator = runtime.has_allocator_accounting
    baseline_allocated = runtime.memory_allocated() if allocator else None
    baseline_reserved = runtime.memory_reserved() if allocator else None
    base_fingerprint = base_fingerprints(pool)
    full_fingerprint = full_fingerprints(pool)
    sweep_interval = max(1, total_steps // 20)

    metrics: list[StepMetric] = []
    violations: list[int] = []
    fingerprints_stable = True
    for step_index, trace_step in enumerate(trace):
        loads_before = pool.stats.loads
        bytes_before = pool.stats.bytes_staged
        started_step = time.perf_counter()
        reservation = pool.acquire_for_step(trace_step.layer_idx, trace_step.expert_ids, storage)
        storage.synchronize()
        elapsed = time.perf_counter() - started_step
        pool.release_step(reservation)

        if allocator and runtime.memory_allocated() != baseline_allocated:
            violations.append(step_index)
        if base_fingerprints(pool) != base_fingerprint:
            fingerprints_stable = False
        if (step_index + 1) % sweep_interval == 0 and full_fingerprints(pool) != full_fingerprint:
            fingerprints_stable = False
        metrics.append(
            StepMetric(
                step=step_index,
                layer_idx=trace_step.layer_idx,
                routed_by=trace_step.routed_by,
                duration_s=elapsed,
                loads=pool.stats.loads - loads_before,
                bytes_moved=pool.stats.bytes_staged - bytes_before,
            )
        )

    if full_fingerprints(pool) != full_fingerprint:
        fingerprints_stable = False
    return _TraceRunResult(
        metrics=metrics,
        allocation_violation_steps=violations,
        fingerprints_stable=fingerprints_stable,
        baseline_allocated=baseline_allocated,
        baseline_reserved=baseline_reserved,
        final_allocated=runtime.memory_allocated() if allocator else None,
        final_reserved=runtime.memory_reserved() if allocator else None,
    )


def _verify_sampled_slots(
    pool: StaticExpertSlotPool,
    storage: AscendPinnedHostStorage,
    trace: Sequence[TraceStep],
    samples: int,
    seed: int,
) -> tuple[bool, int]:
    """Post-loop byte verification (allocates transient device copies on purpose:
    runs strictly after the zero-allocation window has been closed)."""
    rng = random.Random(seed ^ 0xBEEF)
    ok = True
    checked = 0
    for trace_step in rng.sample(list(trace), min(samples, len(trace))):
        for expert_id in trace_step.expert_ids:
            slot = pool.slot_of(trace_step.layer_idx, expert_id)
            if slot == UNRESIDENT_SLOT_ID:
                continue  # valid: a cold long-tail expert may already be evicted
            for param_key, view in pool.param_views(slot).items():
                source = storage.pinned_cpu_weight(trace_step.layer_idx, expert_id, param_key)
                same_device_source = source if source.device == view.device else source.to(view.device)
                if not torch.equal(view, same_device_source):
                    ok = False
                    print(
                        f"verify: MISMATCH layer={trace_step.layer_idx} expert={expert_id} "
                        f"param={param_key} slot={slot}",
                        flush=True,
                    )
                checked += 1
    return ok, checked


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(argv)
    print_plan(config)
    traced_layers, addressable_experts, hash_layers = config.trace_plan()
    probe_simulator = RouterTraceSimulator(
        config.model_config,
        num_experts_available=addressable_experts,
        hot_experts=config.hot_experts,
        hot_ratio=config.hot_ratio,
        zipf_exponent=config.zipf_exponent,
        seed=config.seed,
        num_layers=traced_layers,
        num_hash_layers=hash_layers,
    )
    probe_trace = probe_simulator.generate_trace(config.steps)
    hash_steps = sum(1 for step in probe_trace if step.routed_by == "hash")
    score_steps = len(probe_trace) - hash_steps
    print(
        f"plan: trace OK ({len(probe_trace)} steps, {hash_steps} hash-routed, {score_steps} score-routed)",
        flush=True,
    )
    if config.check_only:
        print("check-only: configuration and capacity plan are valid", flush=True)
        return 0

    report = OffloadStressHarness(config).run()
    print(render_report(report), flush=True)
    if config.report_json is not None:
        config.report_json.parent.mkdir(parents=True, exist_ok=True)
        config.report_json.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
        print(f"report written to {config.report_json}", flush=True)
    return 0 if report.invariants_ok() else 1
