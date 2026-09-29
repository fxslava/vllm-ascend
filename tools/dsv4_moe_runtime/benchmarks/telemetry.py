"""Metrics collection, latency percentiles and stress-report rendering.

Pure reporting (SRP): consumes step metrics produced by a harness and turns
them into tables, warmup curves and JSON-serializable dicts. No orchestration
and no hardware access lives here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..core.slot_pool import StaticExpertSlotPool

GIB = 1024**3
GB = 10**9
BAR_WIDTH = 40


def human_bytes(num_bytes: float) -> str:
    if num_bytes >= GIB:
        return f"{num_bytes / GIB:.2f} GiB"
    if num_bytes >= 2**20:
        return f"{num_bytes / 2**20:.2f} MiB"
    return f"{num_bytes / 1024:.1f} KiB"


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


@dataclass
class StepMetric:
    """One decode-step observation (host wall clock, DMA included)."""

    step: int
    layer_idx: int
    routed_by: str
    duration_s: float
    loads: int
    bytes_moved: int


@dataclass
class StressReport:
    """Aggregate result of one offload stress run."""

    config: object  # BenchConfig (kept untyped here to avoid an import cycle)
    slot_num_bytes: int
    experts_staged: int
    host_staged_bytes: int
    pool_bytes: int
    steps: int
    top_k: int
    baseline_allocated: int | None
    baseline_reserved: int | None
    final_allocated: int | None
    final_reserved: int | None
    allocation_violation_steps: list[int]
    fingerprints_stable: bool
    verification: tuple[bool, int] | None  # (ok, views checked) or None when skipped
    metrics: list[StepMetric]
    wall_time_s: float

    @property
    def allocator_invariant_ok(self) -> bool | None:
        if self.baseline_allocated is None:
            return None
        return not self.allocation_violation_steps

    @property
    def hit_rate_overall(self) -> float:
        activations = self.steps * self.top_k
        hits = sum(self.top_k - metric.loads for metric in self.metrics)
        return hits / activations

    def invariants_ok(self) -> bool:
        verification_ok = self.verification is None or self.verification[0]
        return bool(self.allocator_invariant_ok is not False) and self.fingerprints_stable and verification_ok


def base_fingerprints(pool: StaticExpertSlotPool) -> list[int]:
    """data_ptr of the pool's base tensors (cheap enough for every step)."""
    return [pool.slot_arena.data_ptr(), pool.expert_slot_table.data_ptr(), pool.step_slot_ids_buffer.data_ptr()]


def full_fingerprints(pool: StaticExpertSlotPool) -> list[int]:
    """data_ptr of every base tensor and pre-sliced view the pool owns."""
    tensors = [pool.slot_arena, pool.expert_slot_table, pool.step_slot_ids_buffer]
    for slot_id in range(pool.num_slots):
        tensors.extend(pool.weight_views(slot_id).values())
        tensors.extend(pool.scale_views(slot_id).values())
    return [tensor.data_ptr() for tensor in tensors]


def warmup_curve(report: StressReport, buckets: int) -> list[tuple[str, float]]:
    bucket_size = math.ceil(report.steps / buckets)
    curve: list[tuple[str, float]] = []
    for start in range(0, report.steps, bucket_size):
        window = report.metrics[start : start + bucket_size]
        activations = len(window) * report.top_k
        hits = sum(report.top_k - metric.loads for metric in window)
        curve.append((f"{start:5d}-{start + len(window) - 1:5d}", hits / activations))
    return curve


def latency_percentiles(durations_s: Sequence[float]) -> dict[str, float]:
    milliseconds = [duration * 1000.0 for duration in durations_s]
    return {f"p{int(p)}": percentile(milliseconds, p) for p in (50, 95, 99)}


def bandwidth_summary(report: StressReport) -> dict[str, float | None]:
    miss_metrics = [metric for metric in report.metrics if metric.loads > 0 and metric.duration_s > 0]
    if not miss_metrics:
        return {"aggregate_gbps": None, "p50_gbps": None, "p95_gbps": None}
    total_bytes = sum(metric.bytes_moved for metric in miss_metrics)
    total_time = sum(metric.duration_s for metric in miss_metrics)
    per_step = [metric.bytes_moved / metric.duration_s / GB for metric in miss_metrics]
    return {
        "aggregate_gbps": total_bytes / total_time / GB,
        "p50_gbps": percentile(per_step, 50),
        "p95_gbps": percentile(per_step, 95),
    }


def render_report(report: StressReport) -> str:
    config = report.config  # type: ignore[attr-defined]
    lines: list[str] = []
    separator = "=" * 78
    lines.append(separator)
    lines.append("DeepSeek-V4 NPU offload stress report")
    lines.append(separator)
    lines.append(
        f"device {config.device} | slots {config.pool_slots} x {human_bytes(report.slot_num_bytes)} "
        f"= {human_bytes(report.pool_bytes)} HBM | staged {report.experts_staged} experts "
        f"({human_bytes(report.host_staged_bytes)}, {'pinned' if config.device != 'cpu' else 'host'})"
    )
    lines.append(f"steps {report.steps} (top-{report.top_k}) | wall {report.wall_time_s:.2f}s")

    lines.append("--- cache warmup (hit rate per bucket) ---")
    for label, hit_rate in warmup_curve(report, config.buckets):
        bar = "#" * round(hit_rate * BAR_WIDTH)
        lines.append(f"  steps {label} | {hit_rate:6.1%} | {bar}")
    lines.append(f"  overall hit rate: {report.hit_rate_overall:.1%}")

    lines.append("--- step latency (wall clock: acquire + DMA sync + release) ---")
    for label, subset in (
        ("all steps ", report.metrics),
        ("miss steps", [metric for metric in report.metrics if metric.loads > 0]),
        ("hit steps ", [metric for metric in report.metrics if metric.loads == 0]),
    ):
        stats = latency_percentiles([metric.duration_s for metric in subset])
        lines.append(
            f"  {label}: p50 {stats['p50']:9.3f} ms | p95 {stats['p95']:9.3f} ms | p99 {stats['p99']:9.3f} ms "
            f"(n={len(subset)})"
        )

    lines.append("--- effective miss-transfer bandwidth (host DDR -> HBM) ---")
    bandwidth = bandwidth_summary(report)
    moved = sum(metric.bytes_moved for metric in report.metrics)
    if bandwidth["aggregate_gbps"] is None:
        lines.append("  no miss transfers recorded")
    else:
        lines.append(
            f"  aggregate {bandwidth['aggregate_gbps']:.2f} GB/s | per-step "
            f"p50 {bandwidth['p50_gbps']:.2f} / p95 {bandwidth['p95_gbps']:.2f} GB/s | "
            f"{human_bytes(moved)} moved ({sum(m.loads for m in report.metrics)} expert fills)"
        )

    lines.append("--- zero-allocation invariant ---")
    if report.baseline_allocated is None:
        lines.append("  allocator: n/a on this runtime (cpu); data_ptr fingerprints carry the invariant")
    else:
        verdict = (
            "OK"
            if not report.allocation_violation_steps
            else f"VIOLATED at steps {report.allocation_violation_steps[:10]}"
        )
        lines.append(
            f"  allocator: {verdict} | allocated {human_bytes(report.baseline_allocated)}"
            f" -> {human_bytes(report.final_allocated)} | reserved {human_bytes(report.baseline_reserved)}"
            f" -> {human_bytes(report.final_reserved)}"
        )
    lines.append(f"  fingerprints: {'OK (all data_ptr stable)' if report.fingerprints_stable else 'VIOLATED'}")
    if report.verification is None:
        lines.append("  verification: skipped (--verify-samples 0)")
    else:
        ok, checked = report.verification
        lines.append(f"  verification: {'OK' if ok else 'FAILED'} ({checked} slot views byte-checked)")

    lines.append(f"VERDICT: {'PASS' if report.invariants_ok() else 'FAIL'}")
    lines.append(separator)
    return "\n".join(lines)


def report_to_dict(report: StressReport) -> dict[str, object]:
    bandwidth = bandwidth_summary(report)
    config = report.config  # type: ignore[attr-defined]
    return {
        "config": {
            "device": config.device,
            "steps": report.steps,
            "pool_slots": config.pool_slots,
            "hot_experts": config.hot_experts,
            "hot_ratio": config.hot_ratio,
            "zipf_exponent": config.zipf_exponent,
            "seed": config.seed,
            "small_geometry": config.small_geometry,
            "pinned_layers": config.pinned_layers,
            "pinned_experts_per_layer": config.pinned_experts_per_layer,
        },
        "slot_num_bytes": report.slot_num_bytes,
        "experts_staged": report.experts_staged,
        "host_staged_bytes": report.host_staged_bytes,
        "pool_bytes": report.pool_bytes,
        "hit_rate_overall": report.hit_rate_overall,
        "warmup_curve": [{"steps": label, "hit_rate": rate} for label, rate in warmup_curve(report, config.buckets)],
        "latency_ms": {
            "all": latency_percentiles([m.duration_s for m in report.metrics]),
            "miss": latency_percentiles([m.duration_s for m in report.metrics if m.loads > 0]),
            "hit": latency_percentiles([m.duration_s for m in report.metrics if m.loads == 0]),
        },
        "bandwidth_gbps": bandwidth,
        "total_bytes_moved": sum(m.bytes_moved for m in report.metrics),
        "zero_allocation": {
            "allocator_invariant_ok": report.allocator_invariant_ok,
            "baseline_allocated": report.baseline_allocated,
            "final_allocated": report.final_allocated,
            "baseline_reserved": report.baseline_reserved,
            "final_reserved": report.final_reserved,
            "violation_steps": report.allocation_violation_steps,
            "fingerprints_stable": report.fingerprints_stable,
            "verification": None
            if report.verification is None
            else {"ok": report.verification[0], "views": report.verification[1]},
        },
        "wall_time_s": report.wall_time_s,
        "verdict": "PASS" if report.invariants_ok() else "FAIL",
    }
