"""Production residency & memory-boundary probe.

Three stages, in order:

1. Pre-allocate the contiguous dense backbone workspace (26 GiB on NPU /
   128 MiB CPU mock) and keep it resident -- it models the non-expert
   co-residency (dense weights, KV, activations) the slot pool must tolerate.
2. Grow a monolithic slot-pool allocation batch-by-batch
   (``+probe_slot_batch`` slots) until the allocator raises
   ``torch.OutOfMemoryError``/``RuntimeError``; that boundary is the maximum
   contiguous slot count. Every successful step frees its tensor and releases
   cached blocks so each attempt measures true contiguous capacity.
3. Validate OOM recovery by allocating the *safe production ceiling*
   (``safety_margin`` x discovered maximum) as a real
   :class:`StaticExpertSlotPool`, then run continuous LRU-thrash cycles served
   by :class:`DummyLoopbackWeightProvider` (a single slot-sized pinned host
   buffer, copied non-blocking into slot views) while asserting the
   zero-allocation invariant per cycle.

Library entry point: :func:`run_capacity_probe`. CLI (module invocation only,
it is part of the package)::

    python -m tools.dsv4_moe_runtime.benchmarks.capacity_probe --device npu:0
    python -m dsv4_moe_runtime.benchmarks.capacity_probe --device cpu --small-geometry \
        --backbone-mib 32 --probe-max-slots 384 --thrash-cycles 20
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..benchmarks.capacity_report import (  # re-exported for callers
    CapacityProbeReport,
    ThrashStats,
    render_capacity_report,
)
from ..core.config import SANITY_GEOMETRY, DeepSeekV4MoEConfig
from ..core.layout import ExpertTensorLayout
from ..core.slot_pool import UNRESIDENT_SLOT_ID, StaticExpertSlotPool
from ..hardware.runtime import DeviceRuntime, make_runtime

GIB = 1024**3
MIB = 1024**2
DEFAULT_BACKBONE_NPU_BYTES = 26 * GIB
DEFAULT_BACKBONE_CPU_BYTES = 128 * MIB
DEFAULT_PROBE_SLOT_BATCH = 256
DEFAULT_THRASH_CYCLES = 100
DEFAULT_SLOTS_SWEPT_PER_CYCLE = 256
DEFAULT_SAFETY_MARGIN = 0.9
LOOPBACK_PATTERN_BYTE = 0xA7
LOOPBACK_VERIFY_SAMPLES = 8
VERIFY_SEED_SALT = 0xC1A0

SlotAllocator = Callable[[int, str], torch.Tensor]

__all__ = [
    "CapacityProbeConfig",
    "CapacityProbeReport",
    "DummyLoopbackWeightProvider",
    "ThrashStats",
    "main",
    "render_capacity_report",
    "run_capacity_probe",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepSeek-V4 NPU capacity & residency probe.")
    parser.add_argument("--device", default="npu:0", help="torch device (default npu:0)")
    parser.add_argument("--backbone-mib", type=int, default=None, help="override backbone workspace size in MiB")
    parser.add_argument("--probe-slot-batch", type=int, default=DEFAULT_PROBE_SLOT_BATCH)
    parser.add_argument("--probe-max-slots", type=int, default=None, help="cap growth (recommended on cpu)")
    parser.add_argument("--thrash-cycles", type=int, default=DEFAULT_THRASH_CYCLES)
    parser.add_argument("--slots-swept-per-cycle", type=int, default=DEFAULT_SLOTS_SWEPT_PER_CYCLE)
    parser.add_argument("--safety-margin", type=float, default=DEFAULT_SAFETY_MARGIN)
    parser.add_argument("--small-geometry", action="store_true", help="~13 KiB/slot sanity geometry")
    args = parser.parse_args(argv)
    if not 0.0 < args.safety_margin < 1.0:
        parser.error("--safety-margin must be within (0, 1)")
    if args.probe_slot_batch < 1 or args.thrash_cycles < 1 or args.slots_swept_per_cycle < 1:
        parser.error("batch/cycle/sweep counts must be >= 1")
    config = CapacityProbeConfig(
        device=args.device,
        backbone_bytes=None if args.backbone_mib is None else args.backbone_mib * MIB,
        probe_slot_batch=args.probe_slot_batch,
        probe_max_slots=args.probe_max_slots,
        thrash_cycles=args.thrash_cycles,
        slots_swept_per_cycle=args.slots_swept_per_cycle,
        safety_margin=args.safety_margin,
        small_geometry=args.small_geometry,
    )
    report = run_capacity_probe(config)
    print(render_capacity_report(report), flush=True)
    return 0 if report.passed else 1


@dataclass(frozen=True)
class CapacityProbeConfig:
    """Knobs for one probe run; safe defaults for a 950PR NPU."""

    device: str = "cpu"
    backbone_bytes: int | None = None  # None -> device default (26 GiB NPU / 128 MiB CPU)
    probe_slot_batch: int = DEFAULT_PROBE_SLOT_BATCH
    probe_max_slots: int | None = None  # safety cap for hosts without natural OOM (e.g. cpu)
    thrash_cycles: int = DEFAULT_THRASH_CYCLES
    slots_swept_per_cycle: int = DEFAULT_SLOTS_SWEPT_PER_CYCLE
    safety_margin: float = DEFAULT_SAFETY_MARGIN
    small_geometry: bool = False

    @property
    def model_config(self) -> DeepSeekV4MoEConfig:
        return SANITY_GEOMETRY if self.small_geometry else DeepSeekV4MoEConfig()

    @property
    def layout(self) -> ExpertTensorLayout:
        return ExpertTensorLayout.for_deepseek_v4_flash(self.model_config)

    def resolved_backbone_bytes(self) -> int:
        if self.backbone_bytes is not None:
            return self.backbone_bytes
        return DEFAULT_BACKBONE_CPU_BYTES if self.device == "cpu" else DEFAULT_BACKBONE_NPU_BYTES


class DummyLoopbackWeightProvider:
    """``SlotFillProviderProtocol`` serving one shared slot-sized pinned buffer.

    Loopback semantics: every (layer, expert) maps to the *same* staged bytes
    (a fixed pattern), so this isolates pure transport cost from data
    placement. One host allocation, sliced AOT by the layout; fills are
    non-blocking ``copy_`` on the runtime copy stream.
    """

    def __init__(self, runtime: DeviceRuntime, layout: ExpertTensorLayout, pin: bool | None = None):
        self._runtime = runtime
        self._dma_stream = runtime.make_stream()
        self.bytes_copied = 0
        self.fill_count = 0
        use_pin = runtime.supports_pinned_host_memory if pin is None else pin
        if use_pin and not runtime.supports_pinned_host_memory:
            raise ValueError("pinned host memory requested but unsupported by this runtime")
        self.pinned = use_pin
        arena_flags: dict[str, object] = {"pin_memory": True} if use_pin else {}
        self._arena = torch.empty(layout.slot_num_bytes, dtype=torch.uint8, **arena_flags)
        self._arena.fill_(LOOPBACK_PATTERN_BYTE)
        weights, scales = layout.slice_slot_views(self._arena, 0)
        self._views: dict[str, torch.Tensor] = {}
        for spec in layout.specs:
            source_dict = weights if spec.kind == "packed_fp4" else scales
            self._views[spec.param_key] = source_dict[spec.name]

    def ensure_staged(self, layer_idx: int, expert_id: int) -> None:
        pass  # loopback: every expert is by definition staged

    def pinned_cpu_weight(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor:
        return self._views[param_key]

    def fill_slot_params(self, layer_idx: int, expert_id: int, views) -> int:
        moved = 0
        with self._runtime.stream_context(self._dma_stream):
            for param_key, destination in views.items():
                destination.copy_(self._views[param_key], non_blocking=True)
                moved += destination.numel()
        self.bytes_copied += moved
        self.fill_count += 1
        return moved

    def synchronize(self) -> None:
        self._runtime.synchronize_stream(self._dma_stream)


def _default_allocator(num_bytes: int, device: str) -> torch.Tensor:
    return torch.empty(num_bytes, dtype=torch.uint8, device=device)


def _probe_contiguous_capacity(
    runtime: DeviceRuntime,
    allocate: SlotAllocator,
    slot_bytes: int,
    batch: int,
    max_slots: int | None,
    notes: list[str],
) -> tuple[int, bool, str | None]:
    """Grow a monolithic allocation by ``batch`` slots until the allocator OOMs."""
    allocated_slots = 0
    if max_slots is not None and batch > max_slots:
        batch = max_slots  # a cap below one batch still probes the first batch
    while max_slots is None or allocated_slots + batch <= max_slots:
        try:
            candidate = allocate((allocated_slots + batch) * slot_bytes, runtime.device)
        except (torch.OutOfMemoryError, RuntimeError) as exc:  # OutOfMemoryError subclasses RuntimeError
            runtime.release_cache()
            return allocated_slots, True, f"{type(exc).__name__}: {exc}"
        del candidate  # free before growing so each attempt measures contiguous capacity
        runtime.release_cache()
        allocated_slots += batch
    notes.append(f"probe stopped at the probe-max-slots cap ({max_slots}); no natural OOM observed")
    return allocated_slots, False, None


def _allocate_ceiling_pool(
    config: CapacityProbeConfig,
    layout: ExpertTensorLayout,
    runtime: DeviceRuntime,
    max_contiguous_slots: int,
    notes: list[str],
) -> StaticExpertSlotPool | None:
    """OOM-recovery proof: allocate the safe production ceiling as a real pool."""
    top_k = config.model_config.top_k
    if max_contiguous_slots < top_k:
        notes.append(f"only {max_contiguous_slots} contiguous slots found (below one top-k); skipping ceiling + thrash")
        survivor = torch.empty(layout.slot_num_bytes, dtype=torch.uint8, device=runtime.device)
        del survivor  # the allocator still works after the OOM
        return None
    ceiling_slots = max(top_k, int(max_contiguous_slots * config.safety_margin))
    return StaticExpertSlotPool(config.model_config, ceiling_slots, layout=layout, device=runtime.device)


def _thrash_loop(
    pool: StaticExpertSlotPool,
    runtime: DeviceRuntime,
    provider: DummyLoopbackWeightProvider,
    config: CapacityProbeConfig,
) -> tuple[ThrashStats, list[int], bool]:
    """Continuous full-turnover LRU churn; asserts the zero-alloc invariant."""
    model_config = config.model_config
    top_k = model_config.top_k
    steps_per_cycle = max(1, math.ceil(min(config.slots_swept_per_cycle, pool.num_slots) / top_k))
    accounting = runtime.has_allocator_accounting
    baseline_allocated = runtime.memory_allocated() if accounting else 0
    base_fingerprint = _pool_base_fingerprint(pool)

    violations: list[int] = []
    fingerprints_stable = True
    fills = 0
    hits = 0
    bytes_moved = 0
    steps = 0
    started = time.perf_counter()
    for cycle in range(config.thrash_cycles):
        for step in range(steps_per_cycle):
            layer_idx = (cycle + step) % model_config.num_layers
            key_index = (cycle * steps_per_cycle + step) * top_k
            expert_ids = [(key_index + offset) % model_config.num_routed_experts for offset in range(top_k)]
            loads_before = pool.stats.loads
            bytes_before = pool.stats.bytes_staged
            reservation = pool.acquire_for_step(layer_idx, expert_ids, provider)
            provider.synchronize()
            pool.release_step(reservation)
            misses = pool.stats.loads - loads_before
            fills += misses
            hits += top_k - misses
            bytes_moved += pool.stats.bytes_staged - bytes_before
            steps += 1
            if accounting and runtime.memory_allocated() != baseline_allocated:
                violations.append(cycle)
        if base_fingerprint != _pool_base_fingerprint(pool):
            fingerprints_stable = False
    stats = ThrashStats(
        cycles=config.thrash_cycles,
        steps=steps,
        fills=fills,
        hits=hits,
        bytes_moved=bytes_moved,
        wall_time_s=time.perf_counter() - started,
    )
    return stats, violations, fingerprints_stable


def _pool_base_fingerprint(pool: StaticExpertSlotPool) -> list[int]:
    return [pool.slot_arena.data_ptr(), pool.expert_slot_table.data_ptr(), pool.step_slot_ids_buffer.data_ptr()]


def _verify_loopback_pattern(
    pool: StaticExpertSlotPool,
    provider: DummyLoopbackWeightProvider,
    samples: int,
    seed: int,
) -> tuple[bool, int]:
    """Post-thrash byte check (transient device copies run after the invariant window)."""
    num_layers, num_experts = pool.expert_slot_table.shape
    rng = random.Random(seed ^ VERIFY_SEED_SALT)
    checked = 0
    ok = True
    attempts = 0
    max_attempts = samples * 20
    while checked < samples and attempts < max_attempts:
        attempts += 1
        layer_idx = rng.randrange(num_layers)
        expert_id = rng.randrange(num_experts)
        slot = pool.slot_of(layer_idx, expert_id)
        if slot == UNRESIDENT_SLOT_ID:
            continue
        for param_key, view in pool.param_views(slot).items():
            source = provider.pinned_cpu_weight(layer_idx, expert_id, param_key)
            same_device_source = source if source.device == view.device else source.to(view.device)
            if not torch.equal(view, same_device_source):
                ok = False
            checked += 1
    return ok, checked


def run_capacity_probe(config: CapacityProbeConfig, allocator: SlotAllocator | None = None) -> CapacityProbeReport:
    """Execute all three probe stages and aggregate the report."""
    allocate = allocator or _default_allocator
    layout = config.layout
    slot_bytes = layout.slot_num_bytes
    runtime = make_runtime(config.device)
    notes: list[str] = []

    backbone = allocate(config.resolved_backbone_bytes(), runtime.device)  # Stage 1: stays resident
    runtime.synchronize_device()

    max_slots, oom_detected, oom_error = _probe_contiguous_capacity(  # Stage 2
        runtime, allocate, slot_bytes, config.probe_slot_batch, config.probe_max_slots, notes
    )

    # Stage 3: prove recovery, then stress the safe ceiling under loopback DMA.
    pool: StaticExpertSlotPool | None = None
    thrash: ThrashStats | None = None
    violations: list[int] = []
    fingerprints_stable = True
    loopback_verified: tuple[bool, int] | None = None
    allocated_delta: int | None = None
    oom_recovered = not oom_detected  # nothing to recover from when the probe was capped
    try:
        pool = _allocate_ceiling_pool(config, layout, runtime, max_slots, notes)
        # Either a ceiling pool or the below-top-k survivor allocation succeeded,
        # which proves the allocator came back alive after the OOM.
        oom_recovered = True
        if pool is not None:
            accounting = runtime.has_allocator_accounting
            baseline_allocated = runtime.memory_allocated() if accounting else 0
            provider = DummyLoopbackWeightProvider(runtime, layout)
            thrash, violations, fingerprints_stable = _thrash_loop(pool, runtime, provider, config)
            if accounting:
                allocated_delta = runtime.memory_allocated() - baseline_allocated
            loopback_verified = _verify_loopback_pattern(pool, provider, LOOPBACK_VERIFY_SAMPLES, seed=0)
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        notes.append(f"ceiling allocation failed after OOM: {type(exc).__name__}: {exc}")
        runtime.release_cache()
    del backbone
    runtime.release_cache()

    return CapacityProbeReport(
        device=config.device,
        slot_num_bytes=slot_bytes,
        backbone_bytes=config.resolved_backbone_bytes(),
        probe_slot_batch=config.probe_slot_batch,
        max_contiguous_slots=max_slots,
        oom_detected=oom_detected,
        oom_error=oom_error,
        oom_recovered=oom_recovered,
        ceiling_slots=0 if pool is None else pool.num_slots,
        ceiling_pool_bytes=0 if pool is None else pool.num_slots * slot_bytes,
        thrash=thrash,
        allocation_violation_cycles=violations,
        fingerprints_stable=fingerprints_stable,
        loopback_verified=loopback_verified,
        allocated_delta_bytes=allocated_delta,
        notes=notes,
    )


if __name__ == "__main__":
    sys.exit(main())
