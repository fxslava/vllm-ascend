"""Offload stress harness: config, synthetic-run orchestration and CLI main.

Orchestrates one end-to-end measurement over the memory subsystem only:
acquire -> streamed DMA fill -> stream sync -> release, with a per-step
hardware zero-allocation check. No GEMMs, no attention, no LLM forward.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from ..benchmarks.telemetry import (
    StepMetric,
    StressReport,
    base_fingerprints,
    full_fingerprints,
    render_report,
    report_to_dict,
)
from ..benchmarks.trace_simulator import RouterTraceSimulator, TraceStep
from ..core.config import DeepSeekV4MoEConfig
from ..core.layout import ExpertTensorLayout
from ..core.slot_pool import UNRESIDENT_SLOT_ID, StaticExpertSlotPool
from ..hardware.pinned_storage import AscendPinnedHostStorage
from ..hardware.runtime import DeviceRuntime, make_runtime

DEFAULT_STEPS = 2000
DEFAULT_POOL_SLOTS = 3200
DEFAULT_HOT_EXPERTS = 64
DEFAULT_HOT_RATIO = 0.35
DEFAULT_ZIPF_EXPONENT = 1.2
DEFAULT_PINNED_EXPERTS_PER_LAYER = 256
DEFAULT_BUCKETS = 20

SANITY_GEOMETRY = DeepSeekV4MoEConfig(hidden_size=128, moe_intermediate_size=64, vocab_size=4096)

_NPU_DEFAULTS: dict[str, object] = {
    "device": "npu:0",
    "steps": DEFAULT_STEPS,
    "pool_slots": DEFAULT_POOL_SLOTS,
    "small_geometry": False,
    "pinned_layers": 43,
    "verify_samples": 0,
    "buckets": DEFAULT_BUCKETS,
}
_DRY_RUN_PRESET: dict[str, object] = {
    "device": "cpu",
    "steps": 50,
    "pool_slots": 16,
    "small_geometry": True,
    "pinned_layers": 2,
    "verify_samples": 8,
    "buckets": 5,
}


@dataclass(frozen=True)
class BenchConfig:
    device: str
    steps: int
    pool_slots: int
    hot_experts: int
    hot_ratio: float
    zipf_exponent: float
    seed: int
    small_geometry: bool
    pinned_layers: int
    pinned_experts_per_layer: int
    verify_samples: int
    buckets: int
    report_json: Path | None
    check_only: bool
    pin_host_memory: bool | None = None  # None = auto (pin when the runtime supports it)

    @property
    def model_config(self) -> DeepSeekV4MoEConfig:
        return SANITY_GEOMETRY if self.small_geometry else DeepSeekV4MoEConfig()

    @property
    def layout(self) -> ExpertTensorLayout:
        return ExpertTensorLayout.for_deepseek_v4_flash(self.model_config)

    def trace_plan(self) -> tuple[int, int, int]:
        """(layers traced, experts addressable, hash layers) after staging cuts."""
        model_config = self.model_config
        layers = min(model_config.num_layers, self.pinned_layers)
        experts = min(model_config.num_routed_experts, self.pinned_experts_per_layer)
        return layers, experts, min(model_config.num_hash_layers, layers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepSeek-V4 NPU synthetic offload & memory stress harness (no compute kernels).",
        epilog=(
            "--dry-run forces a small-geometry 50-step CPU sanity run (explicit flags override); "
            "--check-only validates the configuration and capacity plan and exits."
        ),
    )
    parser.add_argument("--device", default=None, help="torch device, e.g. npu:0 (default) or cpu")
    parser.add_argument("--steps", type=int, default=None, help="decode steps (layer acquisitions) to run")
    parser.add_argument("--pool-slots", type=int, default=None, help="HBM expert slots (isomorphic)")
    parser.add_argument("--hot-experts", type=int, default=DEFAULT_HOT_EXPERTS, help="recurring hot head size")
    parser.add_argument(
        "--hot-ratio", type=float, default=DEFAULT_HOT_RATIO, help="share of activations from the hot head"
    )
    parser.add_argument(
        "--zipf-exponent", type=float, default=DEFAULT_ZIPF_EXPONENT, help="Zipf s; higher = more skewed"
    )
    parser.add_argument("--seed", type=int, default=42, help="trace/RNG seed")
    parser.add_argument(
        "--small-geometry", action="store_true", default=None, help="use the ~13 KiB/slot sanity geometry"
    )
    parser.add_argument("--pinned-layers", type=int, default=None, help="host-staged layers (subset for quick runs)")
    parser.add_argument(
        "--pinned-experts-per-layer",
        type=int,
        default=DEFAULT_PINNED_EXPERTS_PER_LAYER,
        help="host-staged experts per layer",
    )
    parser.add_argument(
        "--verify-samples", type=int, default=None, help="post-loop slot views to byte-verify (0 = off)"
    )
    parser.add_argument("--buckets", type=int, default=None, help="warmup-curve buckets")
    parser.add_argument("--report-json", type=Path, default=None, help="write the full report as JSON")
    parser.add_argument("--no-pinned", action="store_true", help="use pageable host memory even if pinning works")
    parser.add_argument("--dry-run", action="store_true", help="short 50-step CPU sanity run (small geometry)")
    parser.add_argument("--check-only", action="store_true", help="validate config + capacity plan, run nothing")
    return parser


def parse_config(argv: Sequence[str] | None = None) -> BenchConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    preset = _DRY_RUN_PRESET if args.dry_run else _NPU_DEFAULTS
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.small_geometry = bool(args.small_geometry)
    top_k = (SANITY_GEOMETRY if args.small_geometry else DeepSeekV4MoEConfig()).top_k

    def fail(condition: bool, message: str) -> None:
        if condition:
            parser.error(message)

    fail(args.steps < 1, "--steps must be >= 1")
    fail(args.pool_slots < top_k, f"--pool-slots must hold a full top-k ({top_k})")
    fail(not 1 <= args.pinned_layers <= 43, "--pinned-layers must be within [1, 43]")
    fail(not 1 <= args.pinned_experts_per_layer <= 256, "--pinned-experts-per-layer must be within [1, 256]")
    fail(
        not 1 <= args.hot_experts <= min(args.pinned_experts_per_layer, 256) - 1,
        "--hot-experts must leave a non-empty cold tail",
    )
    fail(not 0.0 <= args.hot_ratio <= 1.0, "--hot-ratio must be within [0, 1]")
    fail(args.zipf_exponent <= 0.0, "--zipf-exponent must be > 0")
    fail(args.verify_samples < 0, "--verify-samples must be >= 0")
    fail(args.buckets < 1, "--buckets must be >= 1")
    return BenchConfig(
        device=str(args.device),
        steps=args.steps,
        pool_slots=args.pool_slots,
        hot_experts=args.hot_experts,
        hot_ratio=args.hot_ratio,
        zipf_exponent=args.zipf_exponent,
        seed=args.seed,
        small_geometry=args.small_geometry,
        pinned_layers=args.pinned_layers,
        pinned_experts_per_layer=args.pinned_experts_per_layer,
        verify_samples=args.verify_samples,
        buckets=args.buckets,
        report_json=args.report_json,
        check_only=args.check_only,
        pin_host_memory=False if args.no_pinned else None,
    )


def print_plan(config: BenchConfig) -> None:
    layout = config.layout
    model_config = config.model_config
    layers, experts, hash_layers = config.trace_plan()
    pool_bytes = config.pool_slots * layout.slot_num_bytes
    host_bytes = layers * experts * layout.slot_num_bytes
    print(f"plan: device={config.device} layout={'sanity' if config.small_geometry else 'dsv4-flash'}", flush=True)
    print(
        f"      slot={layout.slot_num_bytes / 2**20:.2f} MiB | pool {config.pool_slots} slots "
        f"= {pool_bytes / 2**30:.2f} GiB HBM | staged {layers}x{experts}={layers * experts} experts "
        f"= {host_bytes / 2**30:.2f} GiB host DDR "
        f"({'pinned' if config.device != 'cpu' else 'pin if available'})",
        flush=True,
    )
    print(
        f"      steps={config.steps} (~{config.steps / layers:.1f} tokens x {layers} layers) | "
        f"top-{model_config.top_k} | hot head {config.hot_experts} @ {config.hot_ratio:.0%} | "
        f"zipf s={config.zipf_exponent} | hash layers [0, {hash_layers}) | seed={config.seed}",
        flush=True,
    )


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
