"""Offload stress harness: config, synthetic-run orchestration and CLI main.

Orchestrates one end-to-end measurement of the *exclusive staging* memory
subsystem: acquire -> transit-window stage (streamed on a window miss) ->
DMA fill -> stream sync -> release, with a per-step hardware zero-allocation
check. No GEMMs, no attention, no LLM forward -- and no monolithic host copy:
the only pinned host allocation is the bounded transit window, regardless of
how many layers or steps are simulated.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from ..benchmarks.telemetry import (  # re-exported
    StepMetric,
    StressReport,
    base_fingerprints,
    full_fingerprints,
    human_bytes,
    peak_host_rss_bytes,
    render_report,
    report_to_dict,
)
from ..benchmarks.trace_simulator import RouterTraceSimulator, TraceStep
from ..core.slot_pool import UNRESIDENT_SLOT_ID, StaticExpertSlotPool
from ..hardware.exchange_buffer import TransitExchangeBuffer
from ..hardware.exclusive_staging import ExclusiveStagingProvider
from ..hardware.runtime import DeviceRuntime, make_runtime
from ..hardware.sharded_safetensors import SafetensorsShardIndex, ShardedSafetensorsExpertSource
from ..hardware.vram_budget import assert_budget_fits
from ..protocols.provider import WeightByteSource
from .stress_config import BenchConfig, parse_config, print_plan  # re-exported
from .synthetic_source import SyntheticExpertSource


class OffloadStressHarness:
    """Orchestrates one end-to-end offload measurement (synthetic or real bytes)."""

    def __init__(self, config: BenchConfig, runtime: DeviceRuntime | None = None):
        self._config = config
        self._runtime = runtime

    def run(self) -> StressReport:
        started = time.perf_counter()
        config = self._config
        profile = config.profile
        model_config = config.model_config
        layout = config.layout
        traced_layers, addressable_experts, hash_layers = config.trace_plan()
        first_layer = config.first_moe_layer

        runtime = self._runtime or make_runtime(config.device)
        # Budget before the first device byte: a rejected plan must not leave a
        # half-allocated pool behind, and an actionable message beats an OOM.
        budget = config.device_budget(runtime)
        assert_budget_fits(budget)
        exchange_buffer = TransitExchangeBuffer(
            runtime, layout, host_slots=config.transit_slots, pin=config.pin_host_memory, overflow="drop_oldest"
        )
        source, weight_source_label = _build_byte_source(config, runtime, first_layer, traced_layers)
        provider = ExclusiveStagingProvider(runtime, layout, exchange_buffer, source)
        pool = StaticExpertSlotPool(
            model_config, config.pool_slots, layout=layout, device=runtime.device, exchange_buffer=exchange_buffer
        )
        simulator = RouterTraceSimulator(
            model_config,
            num_experts_available=addressable_experts,
            hot_experts=config.hot_experts,
            hot_ratio=config.hot_ratio,
            zipf_exponent=config.zipf_exponent,
            seed=config.seed,
            num_layers=traced_layers,
            num_hash_layers=hash_layers,
            first_layer_idx=first_layer,
        )
        trace = simulator.generate_trace(config.steps)
        print(
            f"run: {profile.name} exclusive staging -- transit window {config.transit_slots} slots "
            f"({human_bytes(exchange_buffer.capacity_bytes)} pinned cap), pool {pool.num_slots} slots "
            f"({human_bytes(budget.pool_bytes)}), trace {len(trace)} steps over "
            f"{traced_layers}x{addressable_experts} streamable experts from {weight_source_label}",
            flush=True,
        )

        runtime.synchronize_device()
        loop = _execute_trace_loop(pool, provider, exchange_buffer, runtime, trace, config.steps, traced_layers)

        verification = None
        if config.verify_samples:
            verification = _verify_sampled_slots(pool, source, trace, config.verify_samples, config.seed)

        if isinstance(source, ShardedSafetensorsExpertSource):
            source.close()
        return StressReport(
            config=config,
            layout_name=profile.name,
            weight_source=weight_source_label,
            budget=budget,
            slot_num_bytes=layout.slot_num_bytes,
            transit_slots=exchange_buffer.capacity,
            host_window_bytes=exchange_buffer.capacity_bytes,
            transit_peak_in_flight=loop.peak_in_flight,
            window_source_fills=provider.window_source_fills,
            window_hits=provider.window_hits,
            window_refills=provider.window_refills,
            evictions_staged=exchange_buffer.staged_evictions,
            window_dropped=exchange_buffer.dropped_entries,
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
            host_peak_rss_bytes=peak_host_rss_bytes(),
            wall_time_s=time.perf_counter() - started,
        )


def _build_byte_source(
    config: BenchConfig,
    runtime: DeviceRuntime,
    first_layer: int,
    traced_layers: int,
) -> tuple[WeightByteSource, str]:
    """The authoritative expert bytes: real safetensors shards or the seeded pattern."""
    layout = config.layout
    layer_ids = range(first_layer, first_layer + traced_layers)
    num_experts = config.model_config.num_routed_experts
    if config.weights_dir is None:
        return (
            SyntheticExpertSource(
                layout,
                num_layers=traced_layers,
                num_experts=num_experts,
                seed=config.seed,
                first_layer=first_layer,
            ),
            "synthetic expert bytes",
        )
    index = SafetensorsShardIndex.from_directory(config.weights_dir)
    source = ShardedSafetensorsExpertSource(
        runtime,
        index,
        layout,
        layer_ids=layer_ids,
        expert_ids=range(num_experts),
        naming=config.profile.checkpoint_naming,
        strict_alignment=False,  # upstream HF shards align spans to 8 bytes only
    )
    backends = sorted(set(source.backends.values()))
    return source, (
        f"{len(index.shard_paths)} safetensors shards in {config.weights_dir} "
        f"({human_bytes(index.total_bytes)}, io={'+'.join(backends)}, "
        f"{human_bytes(source.chunk_pool_bytes)} {'pinned' if source.pinned_chunks else 'pageable'} chunks)"
    )


@dataclass
class _TraceRunResult:
    """Raw outcome of the measured trace loop (pre-report aggregation)."""

    metrics: list[StepMetric]
    allocation_violation_steps: list[int]
    fingerprints_stable: bool
    peak_in_flight: int
    baseline_allocated: int | None
    baseline_reserved: int | None
    final_allocated: int | None
    final_reserved: int | None


def _execute_trace_loop(
    pool: StaticExpertSlotPool,
    provider: ExclusiveStagingProvider,
    exchange_buffer: TransitExchangeBuffer,
    runtime: DeviceRuntime,
    trace: Sequence[TraceStep],
    total_steps: int,
    layers_per_token: int,
) -> _TraceRunResult:
    """Measured section: acquire -> window stage -> DMA sync -> release.

    Zero-allocation is enforced per step via the runtime allocator (when the
    hardware provides accounting) and via data_ptr fingerprint sweeps over the
    pool AND the transit window (light sweep per step, full view sweep every
    ``total_steps // 20`` steps).
    """
    allocator = runtime.has_allocator_accounting
    baseline_allocated = runtime.memory_allocated() if allocator else None
    baseline_reserved = runtime.memory_reserved() if allocator else None
    window_arena_ptr = exchange_buffer.fingerprint()[0]
    base_fingerprint = base_fingerprints(pool) + [window_arena_ptr]
    full_fingerprint = full_fingerprints(pool) + exchange_buffer.fingerprint()
    sweep_interval = max(1, total_steps // 20)

    metrics: list[StepMetric] = []
    violations: list[int] = []
    fingerprints_stable = True
    peak_in_flight = 0
    for step_index, trace_step in enumerate(trace):
        if step_index and step_index % layers_per_token == 0:
            pool.advance_generation(step_index // layers_per_token - 1)  # token boundary
        loads_before = pool.stats.loads
        bytes_before = pool.stats.bytes_staged
        started_step = time.perf_counter()
        reservation = pool.acquire_for_step(trace_step.layer_idx, trace_step.expert_ids, provider)
        provider.synchronize()
        elapsed = time.perf_counter() - started_step
        pool.release_step(reservation)
        peak_in_flight = max(peak_in_flight, exchange_buffer.in_flight)

        if allocator and runtime.memory_allocated() != baseline_allocated:
            violations.append(step_index)
        if base_fingerprints(pool) + [window_arena_ptr] != base_fingerprint:
            fingerprints_stable = False
        if (step_index + 1) % sweep_interval == 0 and (
            full_fingerprints(pool) + exchange_buffer.fingerprint() != full_fingerprint
        ):
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

    if full_fingerprints(pool) + exchange_buffer.fingerprint() != full_fingerprint:
        fingerprints_stable = False
    return _TraceRunResult(
        metrics=metrics,
        allocation_violation_steps=violations,
        fingerprints_stable=fingerprints_stable,
        peak_in_flight=peak_in_flight,
        baseline_allocated=baseline_allocated,
        baseline_reserved=baseline_reserved,
        final_allocated=runtime.memory_allocated() if allocator else None,
        final_reserved=runtime.memory_reserved() if allocator else None,
    )


def _verify_sampled_slots(
    pool: StaticExpertSlotPool,
    source: WeightByteSource,
    trace: Sequence[TraceStep],
    samples: int,
    seed: int,
) -> tuple[bool, int]:
    """Post-loop byte verification against the authoritative byte source.

    Exclusive staging keeps no host copy, so expected bytes are re-materialized
    from the source (allocates transient copies on purpose: runs strictly after
    the zero-allocation window has been closed).
    """
    rng = random.Random(seed ^ 0xBEEF)
    ok = True
    checked = 0
    for trace_step in rng.sample(list(trace), min(samples, len(trace))):
        for expert_id in trace_step.expert_ids:
            slot = pool.slot_of(trace_step.layer_idx, expert_id)
            if slot == UNRESIDENT_SLOT_ID:
                continue  # valid: a cold long-tail expert may already be evicted
            for param_key, view in pool.param_views(slot).items():
                expected = source.read_param(trace_step.layer_idx, expert_id, param_key)
                same_device_source = expected if expected.device == view.device else expected.to(view.device)
                if not torch.equal(view, same_device_source):
                    ok = False
                    print(
                        f"verify: MISMATCH layer={trace_step.layer_idx} expert={expert_id} "
                        f"param={param_key} slot={slot}",
                        flush=True,
                    )
                checked += 1
    return ok, checked


def _probe_runtime(device: str) -> DeviceRuntime | None:
    """Best-effort runtime for a capacity probe; ``None`` when the device is absent.

    ``--check-only`` must stay runnable on a workstation that has no npu:0, so a
    missing backend downgrades the budget verdict to "unknown" instead of
    failing the plan validation.
    """
    try:
        return make_runtime(device)
    except (RuntimeError, ValueError) as exc:
        print(f"plan: device {device} unavailable for the capacity probe ({exc.__class__.__name__}: {exc})", flush=True)
        return None


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(argv)
    runtime = _probe_runtime(config.device) if config.check_only else make_runtime(config.device)
    print_plan(config, runtime)
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
        first_layer_idx=config.first_moe_layer,
    )
    probe_trace = probe_simulator.generate_trace(config.steps)
    hash_steps = sum(1 for step in probe_trace if step.routed_by == "hash")
    score_steps = len(probe_trace) - hash_steps
    print(
        f"plan: trace OK ({len(probe_trace)} steps, {hash_steps} hash-routed, {score_steps} score-routed)",
        flush=True,
    )
    if config.check_only:
        budget = config.device_budget(runtime)
        if budget.fits is False:
            print(f"check-only: plan does NOT fit the device budget\n{budget.render()}", flush=True)
            return 1
        print("check-only: configuration and capacity plan are valid", flush=True)
        return 0

    report = OffloadStressHarness(config, runtime).run()
    print(render_report(report), flush=True)
    if config.report_json is not None:
        config.report_json.parent.mkdir(parents=True, exist_ok=True)
        config.report_json.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
        print(f"report written to {config.report_json}", flush=True)
    return 0 if report.invariants_ok() else 1
