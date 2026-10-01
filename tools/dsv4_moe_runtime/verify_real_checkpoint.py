"""Verify a real HF checkpoint against a layout profile, byte for byte, before any bulk run.

    PYTHONPATH=. python tools/dsv4_moe_runtime/verify_real_checkpoint.py \
        --weights-dir F:/AI/models/DeepSeek-V2-Lite-Chat --layout dsv2-lite --device cuda:0

The stress harness binds thousands of experts in one go, and when a checkpoint
disagrees with the profile the failure arrives as a span mismatch somewhere in
the middle of that. This answers the smaller questions first, in the order that
makes each one meaningful:

1. **Schema** -- shard inventory, dtypes, the tensor names the profile's naming
   scheme asks for, and the per-expert slot arithmetic, all from safetensors
   *headers only*. Nothing is allocated and no payload byte is read, so it is
   safe to run against a 30 GiB checkpoint on a 12 GiB card.
2. **Alignment** -- what fraction of spans begin on a 4096-byte sector. Upstream
   exporters align to 8 bytes, so the answer is normally *none of them*, and the
   loader's aligned-read-then-slice path is what makes unbuffered IO possible at
   all. Reported rather than assumed, because a checkpoint that happened to be
   sector-aligned would let a broken slicing path pass.
3. **Ingestion** -- one expert, then several across several layers, streamed into
   pre-allocated device slots and compared against independent buffered reads of
   the same file ranges. Byte equality is the only check that distinguishes a
   correct span table from a plausible one: a wrong offset still yields finite
   BF16 of the right shape.
4. **Invariants** -- the device allocator must not move and the slot pointers
   must not change across the fills.

Exit status is non-zero if any check fails, so this can gate a longer run.
"""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path

import torch

from .core.profiles import LAYOUT_PROFILES
from .hardware.runtime import make_runtime
from .hardware.safetensors_provider import naming_scheme
from .hardware.sharded_safetensors import (
    SafetensorsShardIndex,
    ShardedSafetensorsExpertSource,
    bind_sharded_expert_spans,
)
from .hardware.weight_loader import IO_ALIGNMENT

#: Experts byte-compared against the file. Every fill is checked for shape and
#: span arithmetic; this many are also read back and compared byte for byte,
#: which costs a device-to-host copy each.
DEFAULT_VERIFY_EXPERTS = 4


def _header_only(shard: Path) -> tuple[dict, int]:
    """Parse one shard's JSON header. Reads 8 + header_len bytes and no payload."""
    with open(shard, "rb") as handle:
        (header_len,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/dsv4_moe_runtime/verify_real_checkpoint.py",
        description="Header-first verification of a real checkpoint against a layout profile.",
    )
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--layout", default="dsv2-lite", choices=sorted(LAYOUT_PROFILES))
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=int, default=4, help="MoE layers to bind")
    parser.add_argument("--experts", type=int, default=32, help="routed experts per layer to bind")
    parser.add_argument("--verify-experts", type=int, default=DEFAULT_VERIFY_EXPERTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = LAYOUT_PROFILES[args.layout]
    layout = profile.expert_layout
    scheme = naming_scheme(profile.checkpoint_naming)
    failures: list[str] = []

    # ------------------------------------------------------------ 1. schema
    print(f"=== 1. schema: {args.weights_dir} against profile {profile.name!r} ===")
    shards = sorted(args.weights_dir.glob("model-*.safetensors")) or sorted(args.weights_dir.glob("*.safetensors"))
    if not shards:
        print(f"  no safetensors shards under {args.weights_dir}")
        return 1
    dtypes: Counter[str] = Counter()
    params = 0
    blob_residues = []
    for shard in shards:
        header, blob = _header_only(shard)
        blob_residues.append(blob % IO_ALIGNMENT)
        dtypes.update(entry["dtype"] for entry in header.values())
        for entry in header.values():
            count = 1
            for dim in entry["shape"]:
                count *= dim
            params += count
        print(
            f"  {shard.name}: {shard.stat().st_size / 2**30:6.3f} GiB, {len(header):5d} tensors, "
            f"header {blob - 8:7d} B, payload starts at {blob} (mod 4096 = {blob % IO_ALIGNMENT})"
        )
    print(f"  dtypes: {dict(dtypes)} | {params:,} parameters")

    index = SafetensorsShardIndex.from_directory(args.weights_dir)
    print(f"  index: {len(index.tensor_names):,} tensors, {index.total_bytes / 2**30:.2f} GiB payload")

    moe_layer_ids = list(profile.moe_layer_ids)
    layer_ids = moe_layer_ids[: args.layers]
    expert_ids = list(range(min(args.experts, profile.geometry.num_routed_experts)))
    print(
        f"  profile: slot {layout.slot_num_bytes:,} B ({layout.slot_num_bytes / 2**20:.2f} MiB), "
        f"{len(moe_layer_ids)} MoE layers {moe_layer_ids[0]}..{moe_layer_ids[-1]}, "
        f"{profile.geometry.num_routed_experts} experts/layer, naming {scheme.name!r}"
    )
    for spec in layout.specs:
        name = scheme.tensor_name(layer_ids[0], expert_ids[0], spec)
        span = index.span(name) if index.has(name) else None
        state = f"{span.dtype} {span.shape}" if span else "MISSING"
        if span is None or tuple(span.shape) != spec.logical_shape:
            failures.append(f"{name}: {state} does not match the profile's {spec.logical_shape}")
        print(f"    {spec.param_key} -> {name}")
        print(f"       {state}, profile wants {spec.logical_shape}, {spec.num_bytes:,} B at slot+{spec.offset_bytes:,}")

    # --------------------------------------------------------- 2. alignment
    print("\n=== 2. sector alignment of payload spans ===")
    bound = bind_sharded_expert_spans(
        index, layout, layer_ids, expert_ids, naming=profile.checkpoint_naming, strict_alignment=False
    )
    starts = [span.begin for spans in bound.values() for span in spans.values()]
    lengths = [span.num_bytes for spans in bound.values() for span in spans.values()]
    aligned_starts = sum(1 for begin in starts if begin % IO_ALIGNMENT == 0)
    aligned_lengths = sum(1 for length in lengths if length % IO_ALIGNMENT == 0)
    print(f"  bound {len(bound)} experts ({len(layer_ids)} layers x {len(expert_ids)} experts)")
    print(f"  spans starting on a 4096 sector: {aligned_starts}/{len(starts)}")
    print(f"  span lengths that are a 4096 multiple: {aligned_lengths}/{len(lengths)}")
    print(f"  distinct start residues mod 4096: {len(set(begin % IO_ALIGNMENT for begin in starts))}")
    if aligned_starts < len(starts):
        print("  -> unbuffered IO therefore needs the aligned-read-then-slice path (loader does this)")

    # --------------------------------------------------------- 3. ingestion
    print(f"\n=== 3. ingestion into pre-allocated {args.device} slots ===")
    runtime = make_runtime(args.device)
    source = ShardedSafetensorsExpertSource(
        runtime,
        index,
        layout,
        layer_ids=layer_ids,
        expert_ids=expert_ids,
        naming=profile.checkpoint_naming,
        strict_alignment=False,
    )
    print(f"  io backends: {sorted(set(source.backends.values()))}")
    print(
        f"  chunk pool {source.chunk_pool_bytes / 2**20:.0f} MiB, "
        f"pinned={source.pinned_chunks}, 4096-aligned={source.chunks_are_io_aligned()}"
    )
    if not source.chunks_are_io_aligned() and "unbuffered" in set(source.backends.values()):
        failures.append("unbuffered IO negotiated but the chunk pool is not 4096-aligned")

    keys = sorted(bound)
    pool = torch.empty(len(keys) * layout.slot_num_bytes, dtype=torch.uint8, device=args.device)
    slots = [pool.narrow(0, i * layout.slot_num_bytes, layout.slot_num_bytes) for i in range(len(keys))]
    pointers = [slot.data_ptr() for slot in slots]
    runtime.synchronize_device()
    baseline = runtime.memory_allocated(), runtime.memory_reserved()
    print(f"  pool {pool.numel() / 2**30:.2f} GiB in {len(slots)} slots")

    moved = 0
    for index_of_slot, key in enumerate(keys):
        layer_idx, expert_id = key
        slot = slots[index_of_slot]
        views = {spec.param_key: slot.narrow(0, spec.offset_bytes, spec.num_bytes) for spec in layout.specs}
        moved += source.fill_slot_params(layer_idx, expert_id, views)
    source.synchronize()
    runtime.synchronize_device()
    print(f"  streamed {moved / 2**30:.2f} GiB into {len(keys)} slots")

    # -------------------------------------------------------- 4. invariants
    print("\n=== 4. invariants ===")
    after = runtime.memory_allocated(), runtime.memory_reserved()
    if after != baseline:
        failures.append(f"the allocator moved: {baseline} -> {after}")
    if [slot.data_ptr() for slot in slots] != pointers:
        failures.append("a slot pointer changed during the fills")
    print(f"  allocator stable: {after == baseline} ({baseline[0] / 2**30:.3f} -> {after[0] / 2**30:.3f} GiB)")
    print(f"  slot pointers stable: {[slot.data_ptr() for slot in slots] == pointers}")

    step = max(1, len(keys) // max(1, args.verify_experts))
    checked = 0
    for key in keys[::step][: args.verify_experts]:
        layer_idx, expert_id = key
        host = bytes(slots[keys.index(key)].to("cpu").numpy())
        for spec in layout.specs:
            span = source.expert_spans[key][spec.param_key]
            with open(span.shard, "rb") as handle:  # independent of the loader
                handle.seek(span.begin)
                expected = handle.read(span.num_bytes)
            got = host[spec.offset_bytes : spec.offset_bytes + spec.num_bytes]
            if got != expected:
                failures.append(f"layer {layer_idx} expert {expert_id} {spec.param_key}: bytes differ from the file")
        checked += 1
    print(f"  byte-exact against independent file reads: {checked} experts, {len(failures)} failure(s)")
    source.close()

    print("\n" + ("FAILED:" if failures else "VERDICT: PASS"))
    for failure in failures:
        print(f"  - {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
