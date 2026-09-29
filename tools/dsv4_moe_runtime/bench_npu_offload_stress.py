"""NPU synthetic offload & memory stress harness for the DeepSeek-V4 slot pool.

Isolates the memory subsystem from compute: no GEMMs, no attention kernels, no
LLM forward pass. The harness drives ``StaticExpertSlotPool`` on ``npu:0``
through a synthetic MoE router trace and measures the pinned-DDR -> PCIe/MTE3
DMA -> HBM slot pipeline directly:

* ``RouterTraceSimulator`` pre-generates the whole decode trace (AOT): a
  mixture-Zipf activation model with a recurring hot head (~35% of activations)
  and a long cold tail, cycled across layers -- hash-resolved (``tid2eid``) for
  layers 0-2, score top-k emulation for layers 3+.
* ``AscendPinnedHostStorage`` stages mock expert weights in page-locked host
  memory (one monolithic pinned allocation, sliced AOT by
  ``ExpertTensorLayout``) and DMA-copies into slot views with in-place
  ``copy_(..., non_blocking=True)`` on a dedicated copy stream.
* The stress loop asserts the hardware zero-allocation invariant
  (``torch.npu.memory_allocated()`` constant across every step) and stream
  fingerprints (``data_ptr`` stability), and reports step-latency percentiles,
  effective miss-transfer bandwidth and the cache warmup curve.

Usage::

    # Target measurement (Ascend 950PR host, torch_npu + CANN 8.x/9.x):
    python bench_npu_offload_stress.py --device npu:0 --steps 2000 --pool-slots 3200

    # Full-DRAM rehearsal without NPU (needs ~140 GiB host RAM):
    python bench_npu_offload_stress.py --device cpu --steps 500 --pool-slots 512

    # Quick sanity run (small geometry, seconds, any machine):
    python bench_npu_offload_stress.py --dry-run

    # Config/plan validation only, touches no device:
    python bench_npu_offload_stress.py --check-only

Exit codes: 0 pass, 1 invariant violation, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from slot_pool import (
    UNRESIDENT_SLOT_ID,
    DeepSeekV4MoEConfig,
    ExpertTensorLayout,
    StaticExpertSlotPool,
)

GIB = 1024**3
GB = 10**9
USABLE_SLOT_FRACTION = 1.0  # slots are fully usable; kept named for capacity math clarity

DEFAULT_STEPS = 2000
DEFAULT_POOL_SLOTS = 3200
DEFAULT_HOT_EXPERTS = 64
DEFAULT_HOT_RATIO = 0.35
DEFAULT_ZIPF_EXPONENT = 1.2
DEFAULT_PINNED_EXPERTS_PER_LAYER = 256
DEFAULT_BUCKETS = 20
BAR_WIDTH = 40

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


def human_bytes(num_bytes: float) -> str:
    if num_bytes >= GIB:
        return f"{num_bytes / GIB:.2f} GiB"
    if num_bytes >= 2**20:
        return f"{num_bytes / 2**20:.2f} MiB"
    return f"{num_bytes / 1024:.1f} KiB"


# --------------------------------------------------------------------------- #
# Hardware seam: the identical harness runs on npu:0 and on cpu.               #
# --------------------------------------------------------------------------- #


class DeviceRuntime(Protocol):
    """Hardware-facing seam (allocator accounting, copy stream, sync)."""

    device: str
    has_allocator_accounting: bool
    supports_pinned_host_memory: bool

    def memory_allocated(self) -> int: ...

    def memory_reserved(self) -> int: ...

    def synchronize_device(self) -> None: ...

    def make_stream(self) -> object | None: ...

    def stream_context(self, stream: object | None) -> AbstractContextManager[None]: ...

    def synchronize_stream(self, stream: object | None) -> None: ...


class NpuRuntime:
    """Ascend 950PR runtime: torch_npu allocator stats + dedicated DMA stream."""

    def __init__(self, device: str):
        try:
            import torch_npu  # noqa: F401  (registers the torch.npu namespace)
        except ImportError as exc:
            raise RuntimeError(
                f"device {device!r} requires torch_npu (CANN 8.x/9.x, aclnn V5); "
                "run on the Ascend host, or use --device cpu / --dry-run on a workstation"
            ) from exc
        torch.npu.set_device(int(device.rsplit(":", 1)[-1]))
        self.device = device
        self.has_allocator_accounting = True
        self.supports_pinned_host_memory = True

    def memory_allocated(self) -> int:
        return int(torch.npu.memory_allocated(self.device))

    def memory_reserved(self) -> int:
        return int(torch.npu.memory_reserved(self.device))

    def synchronize_device(self) -> None:
        torch.npu.synchronize(self.device)

    def make_stream(self) -> object:
        return torch.npu.Stream()

    def stream_context(self, stream: object | None) -> AbstractContextManager[None]:
        return torch.npu.stream(stream)

    def synchronize_stream(self, stream: object | None) -> None:
        if stream is not None:
            stream.synchronize()


class CpuRuntime:
    """Workstation runtime: no device allocator accounting, no copy stream."""

    def __init__(self, device: str = "cpu"):
        if device != "cpu":
            raise ValueError(f"CpuRuntime serves only 'cpu', got {device!r}")
        self.device = "cpu"
        self.has_allocator_accounting = False
        self.supports_pinned_host_memory = torch.cuda.is_available()

    def memory_allocated(self) -> int:
        return 0

    def memory_reserved(self) -> int:
        return 0

    def synchronize_device(self) -> None:
        pass

    def make_stream(self) -> object | None:
        return None

    def stream_context(self, stream: object | None) -> AbstractContextManager[None]:
        return nullcontext()

    def synchronize_stream(self, stream: object | None) -> None:
        pass


def make_runtime(device: str) -> DeviceRuntime:
    if device == "cpu":
        return CpuRuntime()
    return NpuRuntime(device)


# --------------------------------------------------------------------------- #
# Pinned host DDR staging + dedicated DMA transfer stream.                     #
# --------------------------------------------------------------------------- #


class AscendPinnedHostStorage:
    """Stages every routed expert in page-locked host memory (SlotFillProvider).

    One monolithic pinned allocation is sliced AOT into per-expert parameter
    views via ``ExpertTensorLayout`` -- no host allocation happens at serve
    time either. DMA transfers are in-place
    ``slot_view.copy_(pinned, non_blocking=True)`` on a dedicated copy stream;
    the caller synchronizes via :meth:`synchronize`.
    """

    def __init__(
        self,
        runtime: DeviceRuntime,
        layout: ExpertTensorLayout,
        layer_ids: Sequence[int],
        experts_per_layer: int,
        pin: bool | None = None,
    ):
        self._runtime = runtime
        self._layout = layout
        self._layer_ids = tuple(layer_ids)
        self._experts_per_layer = experts_per_layer
        self._dma_stream = runtime.make_stream()
        self.bytes_copied = 0
        self.dma_copy_count = 0
        use_pin = runtime.supports_pinned_host_memory if pin is None else pin
        if use_pin and not runtime.supports_pinned_host_memory:
            raise ValueError("pinned host memory requested but unsupported by this runtime")
        self.pinned = use_pin
        arena_flags: dict[str, object] = {"pin_memory": True} if use_pin else {}
        self._arena = torch.empty(
            layout.slot_num_bytes * len(self._layer_ids) * experts_per_layer, dtype=torch.uint8, **arena_flags
        )
        self._params: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        for host_slot, layer_idx in enumerate(self._layer_ids):
            for expert_id in range(experts_per_layer):
                weights, scales = layout.slice_slot_views(self._arena, host_slot * experts_per_layer + expert_id)
                views: dict[str, torch.Tensor] = {}
                for spec in layout.specs:
                    source_dict = weights if spec.kind == "packed_fp4" else scales
                    views[spec.param_key] = source_dict[spec.name]
                self._params[(layer_idx, expert_id)] = views

    @property
    def dma_stream(self) -> object | None:
        return self._dma_stream

    @property
    def staged_expert_count(self) -> int:
        return len(self._params)

    def staged_bytes(self) -> int:
        return self._arena.numel()

    def ensure_staged(self, layer_idx: int, expert_id: int) -> None:
        if (layer_idx, expert_id) not in self._params:
            raise KeyError(f"expert (layer={layer_idx}, id={expert_id}) is not staged in host storage")

    def pinned_cpu_weight(self, layer_idx: int, expert_id: int, param_key: str) -> torch.Tensor:
        self.ensure_staged(layer_idx, expert_id)
        return self._params[(layer_idx, expert_id)][param_key]

    def fill_slot_params(self, layer_idx: int, expert_id: int, views: Mapping[str, torch.Tensor]) -> int:
        source_views = self._params[(layer_idx, expert_id)]
        moved = 0
        with self._runtime.stream_context(self._dma_stream):
            for param_key, destination in views.items():
                destination.copy_(source_views[param_key], non_blocking=True)
                moved += destination.numel()
        self.bytes_copied += moved
        self.dma_copy_count += len(views)
        return moved

    def synchronize(self) -> None:
        self._runtime.synchronize_stream(self._dma_stream)


# --------------------------------------------------------------------------- #
# Synthetic router trace: mixture-Zipf activations across 43 layers.           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TraceStep:
    layer_idx: int
    expert_ids: tuple[int, ...]
    routed_by: str  # "hash" (layers 0-2) or "score" (layers 3+)


class RouterTraceSimulator:
    """Pre-generates the full decode trace ahead of the timed loop (AOT).

    Activation model: ``hot_ratio`` of each step's top-k draws come from a
    small recurring hot head, the rest from a long cold tail; both buckets are
    sampled from truncated Zipf weights over popularity ranks (ranks are a
    seeded permutation of expert ids, so popularity is not coupled to id).
    Layers ``[0, num_hash_layers)`` resolve through per-layer ``tid2eid``-style
    tables (expert ids are a pure function of the token id, exactly like the
    checkpoint's hash layers); layers 3+ emulate score top-k output by sampling
    the same mixture (deduplicated to exactly ``top_k`` unique experts).
    """

    def __init__(
        self,
        model_config: DeepSeekV4MoEConfig,
        num_experts_available: int,
        hot_experts: int,
        hot_ratio: float,
        zipf_exponent: float,
        seed: int,
        num_layers: int | None = None,
        num_hash_layers: int | None = None,
    ):
        num_layers = model_config.num_layers if num_layers is None else num_layers
        num_hash_layers = model_config.num_hash_layers if num_hash_layers is None else num_hash_layers
        if not 1 <= hot_experts < num_experts_available:
            raise ValueError(f"hot_experts {hot_experts} outside [1, {num_experts_available})")
        if not 0.0 <= hot_ratio <= 1.0:
            raise ValueError(f"hot_ratio {hot_ratio} outside [0, 1]")
        if num_hash_layers > num_layers:
            raise ValueError(f"{num_hash_layers} hash layers exceed {num_layers} traced layers")
        self._model_config = model_config
        self._num_layers = num_layers
        self._num_hash_layers = num_hash_layers
        self._hot_ratio = hot_ratio

        rng = random.Random(seed ^ 0x5EED_0001)
        universe = list(range(num_experts_available))
        rng.shuffle(universe)  # popularity rank order, decoupled from expert id
        self._hot_ids = universe[:hot_experts]
        self._cold_ids = universe[hot_experts:]
        rank_weight = lambda rank: 1.0 / (rank + 1) ** zipf_exponent  # noqa: E731 (closed over exponent)
        self._hot_cdf = self._cumulative([rank_weight(rank) for rank in range(hot_experts)])
        self._cold_cdf = self._cumulative([rank_weight(rank) for rank in range(hot_experts, num_experts_available)])
        self._seed = seed

    @property
    def hot_expert_ids(self) -> tuple[int, ...]:
        """The recurring hot head (popularity ranks [0, hot_experts))."""
        return tuple(self._hot_ids)

    @staticmethod
    def _cumulative(weights: Sequence[float]) -> tuple[float, ...]:
        total = sum(weights)
        running = 0.0
        cdf: list[float] = []
        for weight in weights:
            running += weight / total
            cdf.append(running)
        return tuple(cdf)

    def _draw(self, ids: Sequence[int], cdf: Sequence[float], rng: random.Random) -> int:
        return ids[min(bisect_right(cdf, rng.random()), len(ids) - 1)]

    def sample_step(self, rng: random.Random) -> tuple[int, ...]:
        """One deduplicated top-k activation set, ordered most-popular first."""
        chosen: list[int] = []
        while len(chosen) < self._model_config.top_k:
            from_hot = rng.random() < self._hot_ratio
            if from_hot:
                expert = self._draw(self._hot_ids, self._hot_cdf, rng)
            else:
                expert = self._draw(self._cold_ids, self._cold_cdf, rng)
            if expert not in chosen:
                chosen.append(expert)
        return tuple(sorted(chosen, key=self._popularity_rank))

    def _popularity_rank(self, expert_id: int) -> int:
        if expert_id in self._hot_ids:
            return self._hot_ids.index(expert_id)
        return len(self._hot_ids) + self._cold_ids.index(expert_id)

    def generate_trace(self, num_steps: int) -> list[TraceStep]:
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        rng = random.Random(self._seed ^ 0xA11CE)
        token_count = math.ceil(num_steps / self._num_layers)
        # tid2eid emulation: row per token, one independent table per hash layer.
        hash_tables: dict[int, list[tuple[int, ...]]] = {
            layer: [tuple(sorted(self.sample_step(rng))) for _ in range(token_count)]
            for layer in range(self._num_hash_layers)
        }
        trace: list[TraceStep] = []
        for step in range(num_steps):
            layer_idx = step % self._num_layers
            token = step // self._num_layers
            if layer_idx < self._num_hash_layers:
                trace.append(TraceStep(layer_idx, hash_tables[layer_idx][token], "hash"))
            else:
                trace.append(TraceStep(layer_idx, self.sample_step(rng), "score"))
        return trace


# --------------------------------------------------------------------------- #
# Configuration & argument parsing.                                            #
# --------------------------------------------------------------------------- #


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
    fail(
        not 1 <= args.pinned_experts_per_layer <= 256,
        "--pinned-experts-per-layer must be within [1, 256]",
    )
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
    pool_bytes = config.pool_slots * layout.slot_num_bytes * USABLE_SLOT_FRACTION
    host_bytes = layers * experts * layout.slot_num_bytes
    print(f"plan: device={config.device} layout={'sanity' if config.small_geometry else 'dsv4-flash'}", flush=True)
    print(
        f"      slot={human_bytes(layout.slot_num_bytes)} | pool {config.pool_slots} slots "
        f"= {human_bytes(pool_bytes)} HBM | staged {layers}x{experts}={layers * experts} experts "
        f"= {human_bytes(host_bytes)} host DDR ({'pinned' if config.device != 'cpu' else 'pin if available'})",
        flush=True,
    )
    print(
        f"      steps={config.steps} (~{config.steps / layers:.1f} tokens x {layers} layers) | "
        f"top-{model_config.top_k} | hot head {config.hot_experts} @ {config.hot_ratio:.0%} | "
        f"zipf s={config.zipf_exponent} | hash layers [0, {hash_layers}) | seed={config.seed}",
        flush=True,
    )


# --------------------------------------------------------------------------- #
# Stress harness.                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class StepMetric:
    step: int
    layer_idx: int
    routed_by: str
    duration_s: float
    loads: int
    bytes_moved: int


@dataclass
class StressReport:
    config: BenchConfig
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


def _base_fingerprints(pool: StaticExpertSlotPool) -> list[int]:
    return [pool.slot_arena.data_ptr(), pool.expert_slot_table.data_ptr(), pool.step_slot_ids_buffer.data_ptr()]


def _full_fingerprints(pool: StaticExpertSlotPool) -> list[int]:
    tensors = [pool.slot_arena, pool.expert_slot_table, pool.step_slot_ids_buffer]
    for slot_id in range(pool.num_slots):
        tensors.extend(pool.weight_views(slot_id).values())
        tensors.extend(pool.scale_views(slot_id).values())
    return [tensor.data_ptr() for tensor in tensors]


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
            f"run: staged {storage.staged_expert_count} experts ({human_bytes(storage.staged_bytes())}, "
            f"{'pinned' if storage.pinned else 'pageable'}), pool {pool.num_slots} slots, "
            f"trace {len(trace)} steps generated",
            flush=True,
        )

        runtime.synchronize_device()
        allocator = runtime.has_allocator_accounting
        baseline_allocated = runtime.memory_allocated() if allocator else None
        baseline_reserved = runtime.memory_reserved() if allocator else None
        base_fingerprints = _base_fingerprints(pool)
        full_fingerprints = _full_fingerprints(pool)

        metrics: list[StepMetric] = []
        violations: list[int] = []
        fingerprints_stable = True
        sweep_interval = max(1, config.steps // 20)
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
            if _base_fingerprints(pool) != base_fingerprints:
                fingerprints_stable = False
            if (step_index + 1) % sweep_interval == 0 and _full_fingerprints(pool) != full_fingerprints:
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

        final_allocated = runtime.memory_allocated() if allocator else None
        final_reserved = runtime.memory_reserved() if allocator else None
        if _full_fingerprints(pool) != full_fingerprints:
            fingerprints_stable = False

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
            baseline_allocated=baseline_allocated,
            baseline_reserved=baseline_reserved,
            final_allocated=final_allocated,
            final_reserved=final_reserved,
            allocation_violation_steps=violations,
            fingerprints_stable=fingerprints_stable,
            verification=verification,
            metrics=metrics,
            wall_time_s=time.perf_counter() - started,
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


def warmup_curve(report: StressReport, buckets: int) -> list[tuple[str, float]]:
    bucket_size = math.ceil(report.steps / buckets)
    curve: list[tuple[str, float]] = []
    for start in range(0, report.steps, bucket_size):
        window = report.metrics[start : start + bucket_size]
        activations = len(window) * report.top_k
        hits = sum(report.top_k - metric.loads for metric in window)
        curve.append((f"{start:5d}-{start + len(window) - 1:5d}", hits / activations))
    return curve


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


def latency_percentiles(durations_s: Sequence[float]) -> dict[str, float]:
    milliseconds = [duration * 1000.0 for duration in durations_s]
    return {f"p{int(p)}": percentile(milliseconds, p) for p in (50, 95, 99)}


def render_report(report: StressReport) -> str:
    config = report.config
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
    return {
        "config": {
            "device": report.config.device,
            "steps": report.steps,
            "pool_slots": report.config.pool_slots,
            "hot_experts": report.config.hot_experts,
            "hot_ratio": report.config.hot_ratio,
            "zipf_exponent": report.config.zipf_exponent,
            "seed": report.config.seed,
            "small_geometry": report.config.small_geometry,
            "pinned_layers": report.config.pinned_layers,
            "pinned_experts_per_layer": report.config.pinned_experts_per_layer,
        },
        "slot_num_bytes": report.slot_num_bytes,
        "experts_staged": report.experts_staged,
        "host_staged_bytes": report.host_staged_bytes,
        "pool_bytes": report.pool_bytes,
        "hit_rate_overall": report.hit_rate_overall,
        "warmup_curve": [
            {"steps": label, "hit_rate": rate} for label, rate in warmup_curve(report, report.config.buckets)
        ],
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


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_config(argv)
    print_plan(config)
    simulator_probe = RouterTraceSimulator(
        config.model_config,
        num_experts_available=config.trace_plan()[1],
        hot_experts=config.hot_experts,
        hot_ratio=config.hot_ratio,
        zipf_exponent=config.zipf_exponent,
        seed=config.seed,
        num_layers=config.trace_plan()[0],
        num_hash_layers=config.trace_plan()[2],
    )
    probe_trace = simulator_probe.generate_trace(config.steps)
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


if __name__ == "__main__":
    sys.exit(main())
