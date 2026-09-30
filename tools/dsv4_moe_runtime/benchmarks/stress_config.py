"""BenchConfig, CLI parsing and the capacity plan print for the stress bench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..core.config import DeepSeekV4MoEConfig, MoEGeometry
from ..core.layout import ExpertTensorLayout
from ..core.profiles import (
    DSV4_FLASH_PROFILE,
    PROFILE_NAMES,
    SANITY_PROFILE,
    MoELayoutProfile,
    describe_profiles,
    profile_for,
)
from ..hardware.runtime import DeviceRuntime
from ..hardware.vram_budget import DeviceMemoryBudget, plan_budget

DEFAULT_STEPS = 2000
DEFAULT_POOL_SLOTS = 3200
DEFAULT_TRANSIT_SLOTS = 2048  # 2048 x 12.75 MiB = 25.49 GiB pinned transit window
MAX_TRANSIT_WINDOW_BYTES = 32 * 1024**3  # hard cap: host staging never exceeds 32 GiB
DEFAULT_HOT_EXPERTS = 64
DEFAULT_HOT_RATIO = 0.35
DEFAULT_ZIPF_EXPONENT = 1.2
DEFAULT_BUCKETS = 20

SANITY_GEOMETRY = DeepSeekV4MoEConfig(hidden_size=128, moe_intermediate_size=64, vocab_size=4096)

# Pool/transit slot counts are per *layout profile* (a 16.50 MiB dsv2-lite slot
# cannot reuse the 12.75 MiB dsv4-flash counts), so the presets below only carry
# the device-and-duration defaults; see MoELayoutProfile.default_pool_slots.
_NPU_DEFAULTS: dict[str, object] = {
    "device": "npu:0",
    "steps": DEFAULT_STEPS,
    "small_geometry": False,
    "verify_samples": 0,
    "buckets": DEFAULT_BUCKETS,
}
_DRY_RUN_PRESET: dict[str, object] = {
    "device": "cpu",
    "steps": 50,
    "pool_slots": 16,
    "small_geometry": True,
    "transit_slots": 64,
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
    transit_slots: int
    verify_samples: int
    buckets: int
    report_json: Path | None
    check_only: bool
    pin_host_memory: bool | None = None  # None = auto (pin when the runtime supports it)
    layout_name: str | None = None  # None = derive from small_geometry (legacy callers)
    weights_dir: Path | None = None  # real safetensors shards instead of synthetic bytes
    count_backbone: bool = True  # charge the resident backbone to the device budget

    @property
    def profile(self) -> MoELayoutProfile:
        """The architecture target of this run (geometry + slot layout + budget)."""
        name = self.layout_name or (SANITY_PROFILE if self.small_geometry else DSV4_FLASH_PROFILE)
        return profile_for(name)

    @property
    def model_config(self) -> MoEGeometry:
        return self.profile.geometry

    @property
    def layout(self) -> ExpertTensorLayout:
        return self.profile.expert_layout

    def trace_plan(self) -> tuple[int, int, int]:
        """(MoE layers traced, experts addressable, hash layers).

        Exclusive staging addressable set is the entire routed-expert space:
        any expert can be streamed through the transit window on demand, so no
        staging cut narrows the trace anymore. Dense prefix layers own no routed
        experts and are skipped (see :attr:`first_moe_layer`).
        """
        profile = self.profile
        layers = profile.num_moe_layers
        return layers, profile.geometry.num_routed_experts, min(profile.geometry.num_hash_layers, layers)

    @property
    def first_moe_layer(self) -> int:
        """Absolute index of the first routed-expert layer (1 on dsv2-lite)."""
        return self.profile.first_moe_layer

    def device_budget(self, runtime: DeviceRuntime | None = None) -> DeviceMemoryBudget:
        """Device-memory admissibility of this plan (capacity probe via ``runtime``)."""
        return plan_budget(
            self.profile,
            pool_slots=self.pool_slots,
            transit_slots=self.transit_slots,
            device=self.device,
            runtime=runtime,
            count_backbone=self.count_backbone,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepSeek-V4 NPU synthetic offload & memory stress harness (no compute kernels).",
        epilog=(
            "--dry-run forces a small-geometry 50-step CPU sanity run (explicit flags override); "
            "--check-only validates the configuration and capacity plan and exits. Host staging is "
            "exclusive: only the bounded --transit-slots window is ever allocated, never all experts. "
            f"Layout profiles -- {describe_profiles()}."
        ),
    )
    parser.add_argument(
        "--device", default=None, help="torch device: npu:0 (default), cuda:0 (RTX 5070 validation) or cpu"
    )
    parser.add_argument(
        "--layout",
        choices=PROFILE_NAMES,
        default=None,
        help=f"architecture profile (default {DSV4_FLASH_PROFILE}); sets geometry, slot bytes and slot defaults",
    )
    parser.add_argument("--steps", type=int, default=None, help="decode steps (layer acquisitions) to run")
    parser.add_argument("--pool-slots", type=int, default=None, help="HBM expert slots (isomorphic)")
    parser.add_argument(
        "--transit-slots",
        type=int,
        default=None,
        help=(
            "pinned host transit window slots (exclusive staging bound; "
            f"default {DEFAULT_TRANSIT_SLOTS} = 25.5 GiB, hard cap 32 GiB)"
        ),
    )
    parser.add_argument(
        "--hot-experts",
        type=int,
        default=None,
        help=f"recurring hot head size (default: the profile's, {DEFAULT_HOT_EXPERTS} on dsv4-flash)",
    )
    parser.add_argument(
        "--hot-ratio", type=float, default=DEFAULT_HOT_RATIO, help="share of activations from the hot head"
    )
    parser.add_argument(
        "--zipf-exponent", type=float, default=DEFAULT_ZIPF_EXPONENT, help="Zipf s; higher = more skewed"
    )
    parser.add_argument("--seed", type=int, default=42, help="trace/RNG seed")
    parser.add_argument(
        "--small-geometry",
        action="store_true",
        default=None,
        help=f"deprecated alias for --layout {SANITY_PROFILE} (~13 KiB/slot sanity geometry)",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="stream real safetensors shards from this directory instead of synthetic expert bytes",
    )
    parser.add_argument(
        "--no-backbone-reserve",
        action="store_true",
        help="budget only this harness's device footprint, not the resident backbone + shared experts",
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
    requested_layout = args.layout
    requested_small_geometry = args.small_geometry is True
    preset = _DRY_RUN_PRESET if args.dry_run else _NPU_DEFAULTS
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    def fail(condition: bool, message: str) -> None:
        if condition:
            parser.error(message)

    # Layout resolution: an explicit --layout wins over the preset's geometry,
    # so `--dry-run --layout dsv2-lite` is a 50-step CPU run of the V2-Lite
    # geometry rather than a contradiction.
    fail(
        requested_layout is not None and requested_small_geometry and requested_layout != SANITY_PROFILE,
        f"--small-geometry is an alias for --layout {SANITY_PROFILE}; it conflicts with --layout {requested_layout}",
    )
    layout_name = requested_layout or (SANITY_PROFILE if bool(args.small_geometry) else DSV4_FLASH_PROFILE)
    profile = profile_for(layout_name)
    args.small_geometry = layout_name == SANITY_PROFILE
    if args.pool_slots is None:
        args.pool_slots = profile.default_pool_slots
    if args.transit_slots is None:
        args.transit_slots = profile.default_transit_slots
    if args.hot_experts is None:
        args.hot_experts = profile.default_hot_experts
    model_config = profile.geometry
    layout = profile.expert_layout
    top_k = model_config.top_k

    fail(args.steps < 1, "--steps must be >= 1")
    fail(args.pool_slots < top_k, f"--pool-slots must hold a full top-k ({top_k})")
    min_transit_slots = 2 * top_k  # a full exchange step: admissions + evictions in flight
    fail(
        args.transit_slots < min_transit_slots,
        f"--transit-slots must hold a full exchange step ({min_transit_slots} = 2 x top-k)",
    )
    fail(
        args.transit_slots * layout.slot_num_bytes > MAX_TRANSIT_WINDOW_BYTES,
        f"--transit-slots x {layout.slot_num_bytes} B/slot exceeds the "
        f"{MAX_TRANSIT_WINDOW_BYTES / 2**30:.0f} GiB host staging cap",
    )
    fail(
        not 1 <= args.hot_experts <= model_config.num_routed_experts - 1,
        "--hot-experts must leave a non-empty cold tail",
    )
    fail(not 0.0 <= args.hot_ratio <= 1.0, "--hot-ratio must be within [0, 1]")
    fail(args.zipf_exponent <= 0.0, "--zipf-exponent must be > 0")
    fail(args.verify_samples < 0, "--verify-samples must be >= 0")
    fail(args.buckets < 1, "--buckets must be >= 1")
    fail(
        args.weights_dir is not None and not args.weights_dir.is_dir(),
        f"--weights-dir {args.weights_dir} is not a directory",
    )
    return BenchConfig(
        device=str(args.device),
        steps=args.steps,
        pool_slots=args.pool_slots,
        hot_experts=args.hot_experts,
        hot_ratio=args.hot_ratio,
        zipf_exponent=args.zipf_exponent,
        seed=args.seed,
        small_geometry=args.small_geometry,
        transit_slots=args.transit_slots,
        verify_samples=args.verify_samples,
        buckets=args.buckets,
        report_json=args.report_json,
        check_only=args.check_only,
        pin_host_memory=False if args.no_pinned else None,
        layout_name=layout_name,
        weights_dir=args.weights_dir,
        count_backbone=not args.no_backbone_reserve,
    )


def print_plan(config: BenchConfig, runtime: DeviceRuntime | None = None) -> None:
    layout = config.layout
    profile = config.profile
    model_config = config.model_config
    layers, experts, hash_layers = config.trace_plan()
    first_layer = config.first_moe_layer
    pool_bytes = config.pool_slots * layout.slot_num_bytes
    window_bytes = config.transit_slots * layout.slot_num_bytes
    print(f"plan: device={config.device} layout={profile.name} ({profile.description})", flush=True)
    print(
        f"      slot={layout.slot_num_bytes / 2**20:.2f} MiB | pool {config.pool_slots} slots "
        f"= {pool_bytes / 2**30:.2f} GiB HBM | transit window {config.transit_slots} slots "
        f"= {window_bytes / 2**30:.2f} GiB pinned host DDR "
        f"(hard cap {MAX_TRANSIT_WINDOW_BYTES / 2**30:.0f} GiB, exclusive staging)",
        flush=True,
    )
    print(
        f"      steps={config.steps} (~{config.steps / layers:.1f} tokens x {layers} MoE layers "
        f"[{first_layer}, {first_layer + layers})) | top-{model_config.top_k} | "
        f"all {layers}x{experts}={layers * experts} experts streamable | "
        f"hot head {config.hot_experts} @ {config.hot_ratio:.0%} | zipf s={config.zipf_exponent} | "
        f"hash layers [0, {hash_layers}) | seed={config.seed}",
        flush=True,
    )
    if profile.num_shared_experts:
        print(
            f"      resident backbone: {profile.num_shared_experts} shared experts/layer + embeddings + MLA "
            f"= {profile.backbone_num_bytes / 2**30:.2f} GiB, never evicted",
            flush=True,
        )
    source = "synthetic expert bytes" if config.weights_dir is None else f"safetensors shards in {config.weights_dir}"
    print(f"      byte source: {source}", flush=True)
    print(f"      {config.device_budget(runtime).render()}", flush=True)
    advisory = profile.pool_slot_advisory
    if advisory is not None and not advisory[0] <= config.pool_slots <= advisory[1]:
        print(
            f"      note: --pool-slots {config.pool_slots} is outside the {profile.name} advisory range "
            f"{advisory[0]}-{advisory[1]} for this target",
            flush=True,
        )
