"""BenchConfig, CLI parsing and the capacity plan print for the stress bench."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..core.config import DeepSeekV4MoEConfig
from ..core.layout import ExpertTensorLayout

DEFAULT_STEPS = 2000
DEFAULT_POOL_SLOTS = 3200
DEFAULT_TRANSIT_SLOTS = 2048  # 2048 x 12.75 MiB = 25.49 GiB pinned transit window
MAX_TRANSIT_WINDOW_BYTES = 32 * 1024**3  # hard cap: host staging never exceeds 32 GiB
DEFAULT_HOT_EXPERTS = 64
DEFAULT_HOT_RATIO = 0.35
DEFAULT_ZIPF_EXPONENT = 1.2
DEFAULT_BUCKETS = 20

SANITY_GEOMETRY = DeepSeekV4MoEConfig(hidden_size=128, moe_intermediate_size=64, vocab_size=4096)

_NPU_DEFAULTS: dict[str, object] = {
    "device": "npu:0",
    "steps": DEFAULT_STEPS,
    "pool_slots": DEFAULT_POOL_SLOTS,
    "small_geometry": False,
    "transit_slots": DEFAULT_TRANSIT_SLOTS,
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

    @property
    def model_config(self) -> DeepSeekV4MoEConfig:
        return SANITY_GEOMETRY if self.small_geometry else DeepSeekV4MoEConfig()

    @property
    def layout(self) -> ExpertTensorLayout:
        return ExpertTensorLayout.for_deepseek_v4_flash(self.model_config)

    def trace_plan(self) -> tuple[int, int, int]:
        """(layers traced, experts addressable, hash layers) -- the full model.

        Exclusive staging addressable set is the entire routed-expert space:
        any expert can be streamed through the transit window on demand, so no
        staging cut narrows the trace anymore.
        """
        model_config = self.model_config
        layers = model_config.num_layers
        return layers, model_config.num_routed_experts, min(model_config.num_hash_layers, layers)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepSeek-V4 NPU synthetic offload & memory stress harness (no compute kernels).",
        epilog=(
            "--dry-run forces a small-geometry 50-step CPU sanity run (explicit flags override); "
            "--check-only validates the configuration and capacity plan and exits. Host staging is "
            "exclusive: only the bounded --transit-slots window is ever allocated, never all experts."
        ),
    )
    parser.add_argument("--device", default=None, help="torch device, e.g. npu:0 (default) or cpu")
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
    model_config = SANITY_GEOMETRY if args.small_geometry else DeepSeekV4MoEConfig()
    layout = ExpertTensorLayout.for_deepseek_v4_flash(model_config)
    top_k = model_config.top_k

    def fail(condition: bool, message: str) -> None:
        if condition:
            parser.error(message)

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
    )


def print_plan(config: BenchConfig) -> None:
    layout = config.layout
    model_config = config.model_config
    layers, experts, hash_layers = config.trace_plan()
    pool_bytes = config.pool_slots * layout.slot_num_bytes
    window_bytes = config.transit_slots * layout.slot_num_bytes
    print(f"plan: device={config.device} layout={'sanity' if config.small_geometry else 'dsv4-flash'}", flush=True)
    print(
        f"      slot={layout.slot_num_bytes / 2**20:.2f} MiB | pool {config.pool_slots} slots "
        f"= {pool_bytes / 2**30:.2f} GiB HBM | transit window {config.transit_slots} slots "
        f"= {window_bytes / 2**30:.2f} GiB pinned host DDR "
        f"(hard cap {MAX_TRANSIT_WINDOW_BYTES / 2**30:.0f} GiB, exclusive staging)",
        flush=True,
    )
    print(
        f"      steps={config.steps} (~{config.steps / layers:.1f} tokens x {layers} layers) | "
        f"top-{model_config.top_k} | all {layers}x{experts}={layers * experts} experts streamable | "
        f"hot head {config.hot_experts} @ {config.hot_ratio:.0%} | zipf s={config.zipf_exponent} | "
        f"hash layers [0, {hash_layers}) | seed={config.seed}",
        flush=True,
    )
