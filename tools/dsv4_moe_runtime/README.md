# DSV4 MoE static runtime (offload harness)

Zero-allocation expert slot pool, generational residency policy, bounded pinned
transit window and DirectStorage-style weight ingestion for offloaded MoE
decode. The same harness runs on Ascend HBM, on a CUDA workstation GPU and on
plain CPU; the device-specific code lives behind one seam
(`hardware/runtime.py`).

## Layout profiles

A profile (`core/profiles.py`) bundles routed-expert geometry, expert slot
storage, checkpoint naming and the memory budget of one target.

| profile | geometry | expert slot | routed working set | notes |
| --- | --- | --- | --- | --- |
| `dsv4-flash` | 43 layers, 256 experts, top-6, hidden 4096 / inter 2048 | 12.75 MiB (block-32 FP4 + E8M0 scales) | 137.1 GiB | production target; hash-routed layers 0-2 |
| `dsv2-lite` | 27 layers (layer 0 dense), 64 experts, top-6, hidden 2048 / inter 1408 | 16.50 MiB (dense BF16, no scales) | 26.8 GiB | DeepSeek-V2-Lite 16B; 2 shared experts/layer resident in the backbone |
| `sanity` | 43 layers, 256 experts at hidden 128 / inter 64 | 13 KiB | 137 MiB | CPU dry runs |

`dsv2-lite` is the local, hardware-identical validation target: 12 GiB of VRAM
against a 26.8 GiB routed working set is the same kind of pressure as 124 GiB of
HBM against DeepSeek-V4 Flash.

Shared experts are *backbone*, not pool: they are activated by every token, so
`BackboneProfile` charges them (with embeddings, MLA projections and the dense
prefix MLP) to device memory permanently and the residency policy never sees
them. For `dsv2-lite` that reservation is 2.44 GiB.

## Devices

`--device cpu | cuda[:i] | npu[:i]`. Each backend supplies allocator
accounting (which carries the zero-allocation invariant), a dedicated copy
stream for H2D DMA, pinned host staging and a capacity probe. CPU has no
allocator accounting, so there the `data_ptr` fingerprint sweeps carry the
invariant alone.

## Memory budget guard

Before the first device byte, `hardware/vram_budget.py` checks

```text
pool_slots x slot_bytes + backbone_bytes + 512 MiB reserve <= free device memory
```

measured after the device context exists. On a 12 GiB RTX 5070 with the
`dsv2-lite` backbone resident that admits **up to 486 pool slots** (6.19 GiB at
the recommended 384); 512 slots does *not* fit, and the guard says so with the
maximum instead of failing later with an OOM. `--no-backbone-reserve` budgets
only the harness footprint. The pinned transit window is host DDR and is never
charged to the device (256 slots = 4.12 GiB on `dsv2-lite`, hard-capped at
32 GiB).

## Benchmark

```bash
PYTHONPATH=. python tools/dsv4_moe_runtime/bench_npu_offload_stress.py \
    --device cuda:0 --layout dsv2-lite --steps 500 --pool-slots 384 \
    --verify-samples 8 --report-json rtx5070_dsv2_lite_report.json
```

Useful flags: `--check-only` (validate the plan and the budget, run nothing),
`--dry-run` (50-step CPU sanity run), `--transit-slots`, `--hot-experts`,
`--no-pinned`, `--weights-dir` (below). The run fails (exit 1) if the
zero-allocation invariant, the `data_ptr` fingerprints, the byte verification,
the host staging cap or the device budget is violated.

## Real weights

`--weights-dir <dir>` replaces the synthetic byte source with an actual
safetensors checkpoint: `hardware/sharded_safetensors.py` parses
`model.safetensors.index.json` (or a single `model.safetensors`), binds every
routed expert's spans across shards with Hugging Face naming
(`model.layers.N.mlp.experts.E.{gate,up,down}_proj.weight`), validates dtype,
shape and per-expert slot totals, then streams each expert NVMe -> pinned host
chunks -> non-blocking H2D DMA. Upstream shards are only 8-byte aligned, so
span binding runs in non-strict alignment mode; the 4096-byte alignment the
unbuffered reads need is absorbed by the fixed pinned chunk pool.

```bash
PYTHONPATH=. python tools/dsv4_moe_runtime/bench_npu_offload_stress.py \
    --device cuda:0 --layout dsv2-lite --steps 500 --pool-slots 384 \
    --weights-dir /path/to/DeepSeek-V2-Lite --verify-samples 8
```

## Real model generation

### Interactive chat

From the repository root, launch the persistent RTX 5070 chat application:

```powershell
python -m tools.dsv4_moe_runtime.chat --slots 384
```

Direct execution with `python tools/dsv4_moe_runtime/chat.py` also works.
Defaults are the local `F:/AI/models/DeepSeek-V2-Lite-Chat` checkpoint, `cuda:0`,
384 routed slots, a 512-token context capacity and at most 64 new tokens per
answer. Override these with `--weights-dir`, `--device`, `--slots`,
`--context-tokens` and `--max-new-tokens`.

The validated decoder now uses a complete exclusive RAM/VRAM partition by
default: all routed experts are loaded at startup, with 384 experts in VRAM
and the remaining 1,280 experts in pinned host RAM. Generation never reopens
checkpoint files. Use `--auto-slots` to size VRAM from the available device
budget after backbone, shared experts, KV storage, transfer scratch and reserve.
Insufficient combined memory fails at startup rather than falling back to disk.
See [the exclusive hierarchy report](benchmarks/EXCLUSIVE_REPORT.md) for details,
the bounded transfer-scratch exception and measured PCIe swap timings.

The model backbone, shared weights and expert pool stay loaded for the whole
session. History uses the checkpoint's native tokenizer template (`User:` and
`Assistant:` in this checkpoint), including BOS and assistant EOS markers.
Each turn resets KV visibility and replays retained history while the expert
cache and monotonic eviction clock persist. Complete oldest conversation turns
are discarded when history would exceed the context capacity; an oversized
single prompt is rejected. Generation uses greedy decoding and streams text,
stopping at the checkpoint EOS token or the configured output limit.

Commands: `/reset` clears history while retaining expert weights; `/stats`
prints routed pool occupancy and cumulative cache counters; `/exit` quits.
Ctrl+C during generation cancels that answer without adding it to history.
Per-answer stats count the full replayed prompt and emitted tokens (including
EOS), wall time including prompt processing, generated tokens/sec over that
wall time, and the turn's routed expert hit rate.

For a non-interactive smoke test:

```powershell
python -m tools.dsv4_moe_runtime.chat --prompt Hello --max-new-tokens 8
```

Repeat `--prompt` to run several conversation turns in one process. The smoke
test on RTX 5070 produced `Hello! How can I help you today`, with 8 prompt
tokens, 8 generated tokens, 9.69 seconds elapsed and a 49.3% routed hit rate.

### Single generation validation

The BF16 V2-Lite decoder loads backbone weights into a fixed arena, reserves
shared experts and KV storage, and executes routed experts through the slot pool.
Attention uses an eager reference path with temporary allocations. See
[the inference audit](INFERENCE_AUDIT.md) for operator boundaries and measured
RTX 5070 results.

```powershell
python -m tools.dsv4_moe_runtime.inference.dsv2_lite --weights-dir F:/AI/models/DeepSeek-V2-Lite-Chat --device cuda:0 --slots 64 --max-new-tokens 8 --report generation.json
```

## Runtime tests

For persistent-pool cold/warm generation measurements, activation footprints
and eviction analysis, see [the cache benchmark report](benchmarks/CACHE_REPORT.md).

```bash
pytest tools/dsv4_moe_runtime/tests/ -v
```

CAT 1-6 cover residency, DMA, memory safety, teardown, real-weight ingestion
and exclusive staging; CAT 7 covers the `dsv2-lite` profile, the budget guard,
sharded ingestion and the CUDA device path (CUDA cases skip without a GPU, the
NPU execution suite skips without an Ascend device).
