"""Dry-run draft inference: single-step decode over synthetic activations.

Validates, without any real compute: shape correctness through every routing /
slot-acquisition / expert-kernel / accumulation stage, and the strict
zero-allocation invariant (``memory_allocated`` unchanged on NPU; every arena
``data_ptr`` constant on every host). The MoE leg runs through the safe
surrogate expert kernel (``hardware/dummy_kernel.py``); CPU runs use the
contract-mock attention backend, and ``npu:0`` runs drive the ctypes V5
wrappers for attention (see ``AclnnV5Backend.fused_attention``). The whole run
executes inside a :class:`RuntimeLifecycleManager` session so DMA streams,
workspaces and arenas are torn down in strict LIFO order even on faults.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..core.config import SANITY_GEOMETRY, DeepSeekV4MoEConfig
from ..core.layout import ExpertTensorLayout
from ..core.slot_pool import StaticExpertSlotPool
from ..draft_inference.backends import AclnnV5Backend, MockV5Backend
from ..draft_inference.engine import DraftInferenceEngine
from ..hardware.dummy_kernel import DEFAULT_EXPERT_LATENCY_US
from ..hardware.lifecycle import RuntimeLifecycleManager
from ..hardware.runtime import make_runtime
from ..protocols.provider import WeightProviderProtocol

DEFAULT_STEPS = 50
DEFAULT_POOL_SLOTS = 32
DEFAULT_ATTENTION_HEADS = 8


@dataclass
class DryRunConfig:
    device: str
    steps: int
    pool_slots: int
    attention_heads: int
    small_geometry: bool
    report_json: Path | None
    expert_latency_us: float = 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draft inference dry run: shapes + zero-allocation over the surrogate expert kernel."
    )
    parser.add_argument("--device", default="npu:0", help="torch device (npu:0 or cpu)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="decode steps to execute")
    parser.add_argument("--pool-slots", type=int, default=DEFAULT_POOL_SLOTS, help="expert slots in HBM")
    parser.add_argument("--attention-heads", type=int, default=DEFAULT_ATTENTION_HEADS)
    parser.add_argument("--small-geometry", action="store_true", help="~13 KiB/slot sanity geometry")
    parser.add_argument(
        "--expert-latency-us",
        type=float,
        default=0.0,
        help=(
            "per-expert surrogate kernel latency in us (0 disables;"
            f" representative band is 150-250, default {DEFAULT_EXPERT_LATENCY_US:g})"
        ),
    )
    parser.add_argument("--report-json", type=Path, default=None, help="write the report as JSON")
    return parser


def parse_dry_run_config(argv: Sequence[str] | None = None) -> DryRunConfig:
    args = build_parser().parse_args(argv)
    if args.steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.pool_slots < 6:
        raise ValueError("--pool-slots must hold a full top-k (6)")
    if args.expert_latency_us < 0.0:
        raise ValueError("--expert-latency-us must be >= 0")
    return DryRunConfig(
        device=args.device,
        steps=args.steps,
        pool_slots=args.pool_slots,
        attention_heads=args.attention_heads,
        small_geometry=args.small_geometry,
        report_json=args.report_json,
        expert_latency_us=args.expert_latency_us,
    )


def build_engine(config: DryRunConfig, provider: WeightProviderProtocol) -> DraftInferenceEngine:
    model_config: DeepSeekV4MoEConfig = SANITY_GEOMETRY if config.small_geometry else DeepSeekV4MoEConfig()
    layout = ExpertTensorLayout.for_deepseek_v4_flash(model_config)
    if config.device == "cpu":
        backend = MockV5Backend()
    else:
        runtime = make_runtime(config.device)
        backend = AclnnV5Backend(runtime, num_heads=config.attention_heads)
    return DraftInferenceEngine(
        config=model_config,
        layout=layout,
        provider=provider,
        backend=backend,
        num_slots=config.pool_slots,
        device=config.device,
        attention_heads=config.attention_heads,
        expert_latency_us=config.expert_latency_us,
    )


def run_draft_dry_run(config: DryRunConfig, provider: WeightProviderProtocol) -> dict[str, object]:
    """Execute the staged steps inside a lifecycle session; return the report."""
    runtime = make_runtime(config.device)
    accounting = runtime.has_allocator_accounting
    engine = build_engine(config, provider)

    lifecycle = RuntimeLifecycleManager(runtime, label="dry-run")
    dma_stream = getattr(provider, "dma_stream", None)
    lifecycle.register_stream("provider-dma-stream", dma_stream)
    lifecycle.register_workspace("v5-backend-workspace", workspace=engine.backend)
    lifecycle.register_arena("slot-pool-arena", arena=engine.pool)

    with lifecycle.session():
        engine.warm_up_routing_tables()
        baseline_allocated = runtime.memory_allocated() if accounting else None
        fingerprint = engine.arena_fingerprint()
        initial_pool_fingerprint = _pool_fingerprint(engine.pool)

        violations: list[int] = []
        totals = {"hits": 0, "misses": 0}
        for step_index in range(config.steps):
            engine.prepare_step(token_id=(step_index * 7) % SANITY_GEOMETRY.vocab_size)
            report = engine.step()
            totals["hits"] += report.layer_hits
            totals["misses"] += report.layer_misses
            if accounting and runtime.memory_allocated() != baseline_allocated:
                violations.append(step_index)
            if engine.arena_fingerprint() != fingerprint:
                raise AssertionError(f"step {step_index}: a scratchpad/pool arena moved")
            if _pool_fingerprint(engine.pool) != initial_pool_fingerprint:
                raise AssertionError(f"step {step_index}: a slot view moved")
        final_allocated = runtime.memory_allocated() if accounting else None

    teardown = lifecycle.last_report
    assert teardown is not None  # the session context guarantees a report

    return {
        "device": config.device,
        "steps": config.steps,
        "layers": (SANITY_GEOMETRY if config.small_geometry else DeepSeekV4MoEConfig()).num_layers,
        "pool_slots": config.pool_slots,
        "attention_backend": "mock" if config.device == "cpu" else "mock (v5 wiring pending bring-up)",
        "moe_gemm_backend": "dummy-surrogate-kernel",
        "expert_kernel": {
            "kind": "dummy-surrogate",
            "latency_us": config.expert_latency_us,
            "executions": getattr(engine.expert_kernel, "executions", 0),
        },
        "layer_hits": totals["hits"],
        "layer_misses": totals["misses"],
        "zero_allocation": {
            "allocator_baseline": baseline_allocated,
            "allocator_final": final_allocated,
            "violations": violations,
            "fingerprints_stable": True,
        },
        "teardown": {
            "trigger": teardown.trigger,
            "lifo_ok": teardown.lifo_order_respected,
            "ok": teardown.ok,
            "steps": [step.name for step in teardown.steps],
            "errors": teardown.errors,
        },
        "verdict": "PASS" if not violations and teardown.ok else "FAIL",
    }


def _pool_fingerprint(pool: StaticExpertSlotPool) -> list[int]:
    tensors = [pool.slot_arena, pool.expert_slot_table, pool.step_slot_ids_buffer]
    for slot_id in range(pool.num_slots):
        tensors.extend(pool.weight_views(slot_id).values())
        tensors.extend(pool.scale_views(slot_id).values())
    return [tensor.data_ptr() for tensor in tensors]


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_dry_run_config(argv)
    from ..hardware.pinned_storage import AscendPinnedHostStorage

    runtime = make_runtime(config.device)
    model_config: DeepSeekV4MoEConfig = SANITY_GEOMETRY if config.small_geometry else DeepSeekV4MoEConfig()
    layout = ExpertTensorLayout.for_deepseek_v4_flash(model_config)
    storage = AscendPinnedHostStorage(
        runtime, layout, layer_ids=range(model_config.num_layers), experts_per_layer=model_config.num_routed_experts
    )
    report = run_draft_dry_run(config, storage)
    print(json.dumps(report, indent=2), flush=True)
    if config.report_json is not None:
        config.report_json.parent.mkdir(parents=True, exist_ok=True)
        config.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if report["verdict"] == "PASS" else 1
