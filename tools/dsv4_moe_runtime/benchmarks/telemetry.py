"""Metrics collection, latency percentiles and stress-report rendering.

Pure reporting (SRP): consumes step metrics produced by a harness and turns
them into tables, warmup curves and JSON-serializable dicts. No orchestration
and no hardware access lives here.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from ..core.slot_pool import StaticExpertSlotPool
from .stress_config import MAX_TRANSIT_WINDOW_BYTES

GIB = 1024**3
GB = 10**9
BAR_WIDTH = 40


def human_bytes(num_bytes: float) -> str:
    if num_bytes >= GIB:
        return f"{num_bytes / GIB:.2f} GiB"
    if num_bytes >= 2**20:
        return f"{num_bytes / 2**20:.2f} MiB"
    return f"{num_bytes / 1024:.1f} KiB"


def peak_host_rss_bytes() -> int | None:
    """Peak host RSS of this process so far; None where unsupported."""
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows has no resource module
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if sys.platform == "darwin" else peak * 1024  # darwin reports bytes, Linux KiB


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
    """Aggregate result of one exclusive-staging offload stress run."""

    config: object  # BenchConfig (kept untyped here to avoid an import cycle)
    slot_num_bytes: int
    transit_slots: int
    host_window_bytes: int
    transit_peak_in_flight: int
    window_source_fills: int
    window_hits: int
    evictions_staged: int
    window_dropped: int
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
    host_peak_rss_bytes: int | None
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

    @property
    def window_cap_ok(self) -> bool:
        """The exclusive staging guarantee: bounded pinned host DDR."""
        return self.host_window_bytes <= MAX_TRANSIT_WINDOW_BYTES

    def invariants_ok(self) -> bool:
        verification_ok = self.verification is None or self.verification[0]
        return (
            bool(self.allocator_invariant_ok is not False)
            and self.fingerprints_stable
            and verification_ok
            and self.window_cap_ok
        )


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
    lines.append("DeepSeek-V4 NPU offload stress report (exclusive staging)")
    lines.append(separator)
    lines.append(
        f"device {config.device} | slots {config.pool_slots} x {human_bytes(report.slot_num_bytes)} "
        f"= {human_bytes(report.pool_bytes)} HBM | exclusive transit window {report.transit_slots} slots "
        f"= {human_bytes(report.host_window_bytes)} pinned host DDR"
    )
    lines.append(
        f"window traffic: {report.window_source_fills} source fills, {report.window_hits} window hits | "
        f"{report.evictions_staged} evictions staged, {report.window_dropped} dropped | "
        f"peak occupancy {report.transit_peak_in_flight}/{report.transit_slots} slots"
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
    rss = "" if report.host_peak_rss_bytes is None else f" | peak process RSS {human_bytes(report.host_peak_rss_bytes)}"
    lines.append(
        f"  host staging: {human_bytes(report.host_window_bytes)} pinned window "
        f"({'within' if report.window_cap_ok else 'OVER'} the "
        f"{human_bytes(MAX_TRANSIT_WINDOW_BYTES)} cap){rss}"
    )
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
            "transit_slots": report.transit_slots,
            "hot_experts": config.hot_experts,
            "hot_ratio": config.hot_ratio,
            "zipf_exponent": config.zipf_exponent,
            "seed": config.seed,
            "small_geometry": config.small_geometry,
        },
        "slot_num_bytes": report.slot_num_bytes,
        "exclusive_staging": {
            "transit_slots": report.transit_slots,
            "host_window_bytes": report.host_window_bytes,
            "window_cap_ok": report.window_cap_ok,
            "window_source_fills": report.window_source_fills,
            "window_hits": report.window_hits,
            "evictions_staged": report.evictions_staged,
            "window_dropped": report.window_dropped,
            "transit_peak_in_flight": report.transit_peak_in_flight,
            "host_peak_rss_bytes": report.host_peak_rss_bytes,
        },
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
