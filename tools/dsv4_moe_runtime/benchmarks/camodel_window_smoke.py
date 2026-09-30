"""CAModel bring-up smoke for the exclusive transit window on npu:0.

The CAModel simulator provides npu:0 without a physical driver (torch_npu
imports with the simulator camodel on LD_LIBRARY_PATH), but it lacks the
aicore op configs for ``torch.full``/``fill_``/``zero_`` on device, so the
:class:`~tools.dsv4_moe_runtime.core.slot_pool.StaticExpertSlotPool` device
tables cannot initialize there. This smoke therefore exercises exactly the
exclusive-staging pieces the simulator CAN run -- the pinned window, keyed
eviction staging (D2H), streamed window fills and promotions (H2D),
consume-on-promote and drop-oldest recycling -- against real device slot
regions allocated with ``torch.empty`` only. The slot pool itself is covered
by the CPU suite.

Run inside the vLLM-Ascend container (repo mounted at /work)::

    SIM=/usr/local/Ascend/cann-9.1.0/x86_64-linux/simulator/dav_3510/camodel
    export LD_LIBRARY_PATH="$SIM:/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64:$LD_LIBRARY_PATH"
    cd /work && python3 -m tools.dsv4_moe_runtime.benchmarks.camodel_window_smoke
"""

from __future__ import annotations

import os
import sys

import torch

from ..core.config import SANITY_GEOMETRY, DeepSeekV4MoEConfig
from ..core.layout import ExpertTensorLayout
from ..hardware.exchange_buffer import TransitExchangeBuffer
from ..hardware.exclusive_staging import ExclusiveStagingProvider
from ..hardware.runtime import NpuRuntime
from .synthetic_source import SyntheticExpertSource

VERIFY_EVERY = 5
PATTERN_STRIDE = 64
PATTERN_BYTE = 0x5A
# CAModel-sized: device ops are emulated and each slot DMA costs minutes of
# simulation, so keep the DMA count low (~25) while still crossing the window
# capacity (fills + evictions > slots) to exercise drop-oldest recycling and
# the post-loop window-hit promotion. CAMODEL_SMOKE_GEOMETRY=micro shrinks the
# slot ~4.7x (hidden=64/inter=32: every param span stays 64-byte divisible).
WINDOW_SLOTS = 8
STEPS = 3
NUM_REGIONS = 1
_GEOMETRIES = {
    "sanity": SANITY_GEOMETRY,
    "micro": DeepSeekV4MoEConfig(hidden_size=64, moe_intermediate_size=32, vocab_size=4096),
}


def _slot_views(region: torch.Tensor, layout: ExpertTensorLayout) -> dict[str, torch.Tensor]:
    """Pool-shaped per-parameter views over one raw slot region."""
    return {
        spec.param_key: region[spec.offset_bytes : spec.offset_bytes + spec.num_bytes].view(spec.view_shape)
        for spec in layout.specs
    }


def _region_matches_source(
    runtime: NpuRuntime, region: torch.Tensor, source: SyntheticExpertSource, layer: int, expert_id: int
) -> bool:
    """D2H a device slot region and byte-compare it against the source tile."""
    back = torch.empty(region.numel(), dtype=torch.uint8, pin_memory=True)
    back.copy_(region, non_blocking=True)
    runtime.synchronize_device()
    expected = source.read_param(layer, expert_id, "w1").reshape(-1)
    return bool(torch.equal(back[: expected.numel()], expected))


def main() -> int:
    geometry = _GEOMETRIES[os.environ.get("CAMODEL_SMOKE_GEOMETRY", "sanity")]
    layout = ExpertTensorLayout.for_deepseek_v4_flash(geometry)
    runtime = NpuRuntime("npu:0")
    window = TransitExchangeBuffer(runtime, layout, host_slots=WINDOW_SLOTS, overflow="drop_oldest")
    source = SyntheticExpertSource(
        layout, num_layers=geometry.num_layers, num_experts=geometry.num_routed_experts, seed=42
    )
    provider = ExclusiveStagingProvider(runtime, layout, window, source)
    print(
        f"window: {window.capacity} slots x {layout.slot_num_bytes} B = "
        f"{window.capacity_bytes / 2**20:.1f} MiB pinned ({window.pinned})",
        flush=True,
    )

    slot_regions = [torch.empty(layout.slot_num_bytes, dtype=torch.uint8, device="npu:0") for _ in range(NUM_REGIONS)]
    resident: dict[int, tuple[int, int]] = {}  # slot region -> key of the bytes it currently holds
    last_evicted: tuple[int, int] | None = None
    top_k = geometry.top_k
    steps = STEPS
    for step in range(steps):
        region_index = step % len(slot_regions)
        layer, expert_base = step % geometry.num_layers, (step * 7) % 200
        expert_ids = [(expert_base + offset) % geometry.num_routed_experts for offset in range(top_k)]
        if region_index in resident:  # exclusive exchange: stage the victim before overwrite
            window.stage_eviction(slot_regions[region_index], key=resident[region_index])
            last_evicted = resident[region_index]
        for expert_id in expert_ids:
            provider.ensure_staged(layer, expert_id)
            provider.fill_slot_params(layer, expert_id, _slot_views(slot_regions[region_index], layout))
        provider.synchronize()
        resident[region_index] = (layer, expert_ids[-1])
        if step % VERIFY_EVERY == 0 and not _region_matches_source(
            runtime, slot_regions[region_index], source, layer, expert_ids[-1]
        ):
            print(f"step {step}: HBM bytes do not match the source tile", flush=True)
            return 1
        if step % 30 == 0:
            print(
                f"step {step}: fills {provider.window_source_fills}, hits {provider.window_hits}, "
                f"dropped {window.dropped_entries}, in-flight {window.in_flight}",
                flush=True,
            )

    if last_evicted is None or window.dropped_entries == 0:
        print("churn did not exercise eviction staging or drop-oldest recycling", flush=True)
        return 1

    # Window-hit path: the most recently evicted key must still live in the
    # window and promote back with byte-exact content (no source refill).
    fills_before = provider.window_source_fills
    provider.ensure_staged(*last_evicted)
    if provider.window_hits == 0:
        print(f"key {last_evicted} missed the window; refill happened", flush=True)
        return 1
    probe = torch.empty(layout.slot_num_bytes, dtype=torch.uint8, device="npu:0")
    provider.fill_slot_params(*last_evicted, _slot_views(probe, layout))
    provider.synchronize()
    if provider.window_source_fills != fills_before or not _region_matches_source(
        runtime, probe, source, *last_evicted
    ):
        print(f"window-hit promotion for {last_evicted} was not byte-exact", flush=True)
        return 1

    print(
        f"done: {steps} steps | source fills {provider.window_source_fills}, window hits "
        f"{provider.window_hits}, evictions staged {window.staged_evictions}, dropped "
        f"{window.dropped_entries} (window never exceeded {window.capacity} slots)",
        flush=True,
    )
    import resource

    print(f"peak host RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2**20:.0f} MiB", flush=True)
    print("CAMODEL WINDOW SMOKE PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
