"""Device memory budget guard: does this plan fit in the target's VRAM/HBM?

The offload runtime commits device memory in exactly two places -- the static
expert slot pool and the always-resident backbone (embeddings, MLA projections,
dense prefix layers, shared experts) -- plus a pinned *host* transit window that
never touches the device. A plan is admissible when

    pool_bytes + backbone_bytes + DEVICE_WORKSPACE_RESERVE_BYTES <= free VRAM

measured *after* the device context exists, so the driver/context footprint is
already excluded from ``free`` and is never double-counted.

The reserve covers what the harness does not allocate itself but a real decode
step will: kernel workspaces, activations, KV cache growth and caching-allocator
fragmentation. It is deliberately a single named constant instead of a
percentage, so the same guard reads identically on a 12 GiB RTX 5070 and on
124 GiB of Ascend HBM.

``check_only`` runs and dry runs may have no capacity numbers at all (CPU
runtime, or a backend without ``mem_get_info``): the budget then reports
``fits=None`` -- unknown, not OK -- and the caller prints it without failing.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.profiles import MoELayoutProfile
from .runtime import DeviceRuntime

#: Headroom left to kernel workspaces, activations and allocator fragmentation.
DEVICE_WORKSPACE_RESERVE_BYTES = 512 * 1024**2

GIB = 1024**3


class MemoryBudgetError(RuntimeError):
    """Raised when a plan cannot fit the target device's memory."""


@dataclass(frozen=True)
class DeviceMemoryBudget:
    """One admissibility verdict for a (device, pool, backbone) triple."""

    device: str
    slot_num_bytes: int
    pool_slots: int
    transit_slots: int
    backbone_bytes: int
    backbone_modeled: bool
    reserve_bytes: int
    total_bytes: int | None
    free_bytes: int | None

    @property
    def pool_bytes(self) -> int:
        return self.pool_slots * self.slot_num_bytes

    @property
    def transit_host_bytes(self) -> int:
        """Pinned host DDR: the window is host-side and never charged to VRAM."""
        return self.transit_slots * self.slot_num_bytes

    @property
    def harness_device_bytes(self) -> int:
        """What this benchmark actually allocates on the device."""
        return self.pool_bytes

    @property
    def deployment_device_bytes(self) -> int:
        """What the full runtime holds: slot pool plus the resident backbone."""
        return self.pool_bytes + self.backbone_bytes

    @property
    def required_bytes(self) -> int:
        return self.deployment_device_bytes + self.reserve_bytes

    @property
    def fits(self) -> bool | None:
        """``True``/``False`` against free device memory, ``None`` if unknown."""
        if self.free_bytes is None:
            return None
        return self.required_bytes <= self.free_bytes

    @property
    def headroom_bytes(self) -> int | None:
        if self.free_bytes is None:
            return None
        return self.free_bytes - self.required_bytes

    @property
    def max_pool_slots(self) -> int | None:
        """Largest pool that would still fit (same backbone and reserve)."""
        if self.free_bytes is None:
            return None
        spare = self.free_bytes - self.backbone_bytes - self.reserve_bytes
        return max(spare // self.slot_num_bytes, 0)

    def to_dict(self) -> dict[str, object]:
        return {
            "device": self.device,
            "slot_num_bytes": self.slot_num_bytes,
            "pool_slots": self.pool_slots,
            "pool_bytes": self.pool_bytes,
            "backbone_bytes": self.backbone_bytes,
            "backbone_modeled": self.backbone_modeled,
            "transit_host_bytes": self.transit_host_bytes,
            "reserve_bytes": self.reserve_bytes,
            "deployment_device_bytes": self.deployment_device_bytes,
            "device_total_bytes": self.total_bytes,
            "device_free_bytes": self.free_bytes,
            "fits": self.fits,
            "headroom_bytes": self.headroom_bytes,
            "max_pool_slots": self.max_pool_slots,
        }

    def render(self) -> str:
        backbone = (
            f"{self.backbone_bytes / GIB:.2f} GiB backbone+shared"
            if self.backbone_modeled
            else "backbone not modeled for this profile"
        )
        capacity = (
            "device capacity unknown"
            if self.free_bytes is None
            else f"{self.free_bytes / GIB:.2f} GiB free of {(self.total_bytes or 0) / GIB:.2f} GiB"
        )
        verdict = {True: "FITS", False: "OVER BUDGET", None: "UNKNOWN"}[self.fits]
        line = (
            f"budget[{self.device}]: {self.pool_slots} slots x {self.slot_num_bytes / 2**20:.2f} MiB "
            f"= {self.pool_bytes / GIB:.2f} GiB pool + {backbone} + "
            f"{self.reserve_bytes / GIB:.2f} GiB reserve = {self.required_bytes / GIB:.2f} GiB "
            f"vs {capacity} -> {verdict}"
        )
        if self.max_pool_slots is not None:
            line += f" (max {self.max_pool_slots} slots)"
        return line


def plan_budget(
    profile: MoELayoutProfile,
    pool_slots: int,
    transit_slots: int,
    device: str,
    runtime: DeviceRuntime | None = None,
    count_backbone: bool = True,
    reserve_bytes: int = DEVICE_WORKSPACE_RESERVE_BYTES,
) -> DeviceMemoryBudget:
    """Build the budget for a plan; ``runtime`` supplies the capacity probe.

    Without a runtime (parse/plan time) the capacity is unknown and the verdict
    is ``None``; the same call with a live runtime turns it into a hard verdict.
    """
    backbone_modeled = count_backbone and profile.backbone is not None
    return DeviceMemoryBudget(
        device=device,
        slot_num_bytes=profile.expert_layout.slot_num_bytes,
        pool_slots=pool_slots,
        transit_slots=transit_slots,
        backbone_bytes=profile.backbone_num_bytes if backbone_modeled else 0,
        backbone_modeled=backbone_modeled,
        reserve_bytes=reserve_bytes,
        total_bytes=None if runtime is None else runtime.device_total_memory(),
        free_bytes=None if runtime is None else runtime.device_free_memory(),
    )


def assert_budget_fits(budget: DeviceMemoryBudget) -> None:
    """Fail fast with an actionable message instead of a mid-run device OOM."""
    if budget.fits is not False:
        return
    raise MemoryBudgetError(
        f"{budget.render()}\n"
        f"  reduce --pool-slots to <= {budget.max_pool_slots} (currently {budget.pool_slots}), "
        f"free device memory, or pass --no-backbone-reserve to budget the harness footprint only"
    )
