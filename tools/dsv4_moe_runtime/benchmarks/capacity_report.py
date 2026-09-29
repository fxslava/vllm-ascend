"""Capacity probe report model, rendering and CLI entry point."""

from __future__ import annotations

from dataclasses import dataclass, field

from .telemetry import human_bytes


@dataclass
class ThrashStats:
    cycles: int
    steps: int
    fills: int
    hits: int
    bytes_moved: int
    wall_time_s: float


@dataclass
class CapacityProbeReport:
    """Aggregate outcome of one probe run (stages 1-3)."""

    device: str
    slot_num_bytes: int
    backbone_bytes: int
    probe_slot_batch: int
    max_contiguous_slots: int
    oom_detected: bool
    oom_error: str | None
    oom_recovered: bool
    ceiling_slots: int
    ceiling_pool_bytes: int
    thrash: ThrashStats | None
    allocation_violation_cycles: list[int] = field(default_factory=list)
    fingerprints_stable: bool = True
    loopback_verified: tuple[bool, int] | None = None
    allocated_delta_bytes: int | None = None  # None when the runtime has no allocator accounting
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        if self.oom_detected and not self.oom_recovered:
            return False
        if self.allocation_violation_cycles or not self.fingerprints_stable:
            return False
        if self.loopback_verified is not None and not self.loopback_verified[0]:
            return False
        return self.allocated_delta_bytes in (None, 0)


def render_capacity_report(report: CapacityProbeReport) -> str:
    separator = "=" * 78
    lines = [
        separator,
        "DeepSeek-V4 capacity & residency probe report",
        separator,
        f"device {report.device} | slot {human_bytes(report.slot_num_bytes)} | "
        f"backbone {human_bytes(report.backbone_bytes)} resident",
        (
            f"probe: +{report.probe_slot_batch} slots/step -> max contiguous {report.max_contiguous_slots} slots "
            f"({human_bytes(report.max_contiguous_slots * report.slot_num_bytes)}) | "
            f"OOM {'reached (' + report.oom_error + ')' if report.oom_detected else 'not reached (capped)'}"
        ),
        (
            f"recovery: {'OK' if report.oom_recovered else 'FAILED'} | safe ceiling "
            f"{report.ceiling_slots} slots ({human_bytes(report.ceiling_pool_bytes)})"
        ),
    ]
    if report.thrash is None:
        lines.append("thrash: skipped")
    else:
        lines.append(
            f"thrash: {report.thrash.cycles} cycles, {report.thrash.steps} steps, {report.thrash.fills} fills "
            f"({report.thrash.hits} hits), {human_bytes(report.thrash.bytes_moved)} moved, "
            f"{report.thrash.wall_time_s:.2f}s"
        )
    allocator_line = (
        f"allocator delta {report.allocated_delta_bytes} bytes"
        if report.allocated_delta_bytes is not None
        else "allocator n/a on this runtime (cpu); data_ptr fingerprints carry the invariant"
    )
    lines.append(
        f"zero-allocation: {allocator_line} | fingerprints: {'OK' if report.fingerprints_stable else 'VIOLATED'}"
    )
    if report.loopback_verified is None:
        lines.append("loopback verification: skipped")
    else:
        ok, checked = report.loopback_verified
        lines.append(f"loopback verification: {'OK' if ok else 'FAILED'} ({checked} slot views byte-checked)")
    for note in report.notes:
        lines.append(f"note: {note}")
    lines.append(f"VERDICT: {'PASS' if report.passed else 'FAIL'}")
    lines.append(separator)
    return "\n".join(lines)
