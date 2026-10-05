# Expert cache benchmark

Measured on the RTX 5070 using the real DeepSeek-V2-Lite-Chat BF16 checkpoint
on the Samsung SSD 970 EVO Plus 1TB NVMe drive mounted as F:.
Each configuration keeps one decoder and expert pool alive for three identical
generations. Sequence resets hide old KV entries while retaining weights and
advancing the eviction policy's token clock monotonically.

## Results

Mean decode-step wall time, excluding prompt processing:

| Workload / routed slots | Run-1 cold | Run-2 replay | Run-3 replay | Run-2 misses / hit rate | Run-2 streamed payload |
| --- | --- | --- | --- | --- | --- |
| Chat Hello, 8 generated tokens / 64 | 1.026 s | 1.067 s | 1.340 s | 2073 / 11.4% | 33.40 GiB |
| Same chat / 156 | 0.921 s | 1.496 s | 0.897 s | 1981 / 15.3% | 31.92 GiB |
| Same chat / 384 | 0.521 s | 0.505 s | 0.514 s | 1123 / 52.0% | 18.10 GiB |
| Plain Hello, 3 generated tokens / 373 | 0.862 s | 0.0503 s | 0.0511 s | 0 / 100% | 0 |

The fitting workload processes one prompt token and two generated tokens and
outputs `, I am`. The chat workload processes eight prompt tokens and seven
generated tokens and outputs `Hello! How can I help you today`. The final
generated token is selected from logits and is not passed through another
forward step. Output token IDs match across all three runs in each configuration.

Exactly 373 slots retain the fitting workload's union: both warm runs have
468 hits, zero misses, zero evictions and zero file-read requests. A 480-slot
control also gives zero streaming and approximately 48-49 ms mean warm decode.
The 373-slot same-workload speedup is approximately 17x. Compared with the earlier
1.03-second chat baseline it is approximately 20x, but those contexts differ.
Larger pools alone do not eliminate loading for an over-capacity workload.

## Cold versus warm breakdown

All three processed tokens of the fitting 373-slot workload:

| Category | Run-1 | Run-2 | Run-3 |
| --- | --- | --- | --- |
| Total generation forward time | 3.121107 s | 0.151326 s | 0.153086 s |
| Routed weight fill service: file reads + H2D completion | 2.711887 s | 0 | 0 |
| Pool acquisition overhead excluding fills | 0.074347 s | 0.046698 s | 0.046746 s |
| Remaining execution and host orchestration | 0.334872 s | 0.104628 s | 0.106340 s |
| Misses / hits / evictions | 373 / 95 / 0 | 0 / 468 / 0 | 0 / 468 / 0 |
| Routed payload bytes | 6,453,460,992 | 0 | 0 |
| File-read request bytes including alignment | 6,455,835,296 | 0 | 0 |

The components partition synchronized wall time. The last category includes
attention, RMSNorm, routing, expert GEMMs, residuals, the LM head, host launches
and synchronization; it is not pure GPU kernel time. Fill service combines IO
and DMA, which the streaming loader can overlap internally. Initialization of
the backbone and 26 shared regions is excluded from these generation counters.

Cold means an empty routed VRAM cache, not flushed filesystem or SSD caches.
The actual loaders negotiated Windows unbuffered IO. Read-request bytes count
the reader API's aligned reads and buffered tails; they do not measure SSD NAND
traffic. The loader's existing `bytes_read` counter counts logical valid bytes,
not padded sectors. The newer 384/373/480 reports additionally record reader
request bytes. Timings include per-layer synchronization for measurement; these
are short observations, not statistically controlled hardware benchmarks.

## Activation footprint and hit transition

Expert identity is `(layer, expert_id)`. Every token therefore requests exactly
156 different matrices: six in each of 26 layers. Expert number 7 in two layers
represents different weights and cannot share one slot. Clustering affects reuse
between tokens within a layer, not the count of 156 per-token requests.

The chat trace's cumulative unique keys at t0 through t14 are:

```text
156, 282, 382, 445, 527, 561, 616, 641, 673, 691, 752, 811, 851, 900, 916
```

Its union requires 14.76 GiB of routed slots alone, before the 2.44 GiB
backbone/shared weights and scratch. It cannot remain fully resident on this
12 GiB card. Pre-warming frequent experts cannot remove all file reads while
this union exceeds capacity. A semantically similar prompt is not guaranteed to
reuse the same experts; this benchmark intentionally uses exact replay.

| Fitting trace position | Cold hits / misses | Cold hit rate | Warm hits / misses | Warm hit rate | Cumulative unique |
| --- | --- | --- | --- | --- | --- |
| t0, plain Hello | 0 / 156 | 0% | 156 / 0 | 100% | 156 |
| t1, comma | 18 / 138 | 11.5% | 156 / 0 | 100% | 294 |
| t2, I | 77 / 79 | 49.4% | 156 / 0 | 100% | 373 |

384-slot chat hit rates at t0 through t14:

```text
Cold:  0.0, 19.2, 35.9, 59.6, 35.9, 75.6, 39.7, 62.8, 51.9, 75.6, 44.9, 51.3, 66.0, 49.4, 71.8 percent
Warm: 23.1, 35.3, 37.2, 59.6, 35.9, 75.6, 39.7, 62.8, 51.9, 75.6, 44.9, 51.3, 66.0, 49.4, 71.8 percent
```

## Residency and eviction audit

- The pool allocates and zeros the entire arena at construction. Routed slots
  are 16.5 MiB each; two shared slots per MoE layer occupy 858 MiB permanently.
  Loading a slot changes occupancy and bytes, not its VRAM allocation.
- A lookup in the resident mapping decides hit/miss. Hits increment `stats.hits`
  without calling the provider. Missing keys consume free slots, then evict
  eligible residents. Successful fills increment `stats.loads` and staged bytes.
- Requested top-k and locked slots are excluded from eviction. The policy ranks
  older generations first, then lower frequency and older access. Generation
  saturates at two and inactive frequencies decay at token completion. There
  are no hash-layer immunity rules active for V2-Lite.
- Locks last for one layer's expert execution, not a whole token. Later layers
  may evict earlier layers' experts, explaining why 156 slots do not retain the
  multi-token trace. Resident mappings persist across tokens and sequences.
- This decoder has no transit exchange buffer. Evicted weights are reloaded
  from the authoritative checkpoint if requested again.
- The advertised acquisition transaction covers admission planning/source
  validation. A provider failure during a physical fill occurs after policy and
  residency metadata commit; it does not fully roll back partially written
  slots. No such failure occurred in these runs.
- Fixed sequence reset preserves residency and the global completed-token count.
  Rewinding that count would make `advance_generation` ignore subsequent
  boundaries until the old high-water mark was exceeded.

For 373 routed slots, the arena is 6.848 GiB both empty and populated: **arena
VRAM delta is zero**. Routed valid occupancy grows from zero to 6,453,460,992
bytes. Total Torch live allocation grows from 8.455 to 8.487 GiB (32.195 MiB);
that additional allocation comes from execution buffers/retained output, not
slot growth. Arena and scratch fingerprints stay fixed. Cached weights reside
in VRAM; this test does not establish persistent residency in GPU SRAM.

## Reproduction and evidence

From the repository root:

```powershell
python -m tools.dsv4_moe_runtime.benchmarks.cache_generation --weights-dir F:/AI/models/DeepSeek-V2-Lite-Chat --slots 64 156 384 --report cache_chat.json
python -m tools.dsv4_moe_runtime.benchmarks.cache_generation --weights-dir F:/AI/models/DeepSeek-V2-Lite-Chat --slots 373 --raw-prompt --new-tokens 3 --report cache_fitting.json
```

[64/156-slot trace](cache_generation.json), [384-slot trace](cache_chat384.json),
[exact-fit 373-slot trace](cache_fitting373.json),
[480-slot control](cache_fitting480.json). Each contains per-token timings,
hits/misses/evictions, routes, cumulative footprint, per-layer unique counts,
frequency histograms and residency counters. The 384-slot case was measured
in a separate process after the initial sweep encountered retained CUDA allocator
reservation; benchmark teardown now releases cached allocator blocks between
configurations.

Validation: 131 runtime tests passed, 10 Ascend tests skipped; changed Python
files pass Ruff. The existing read-only mmap warning remains.
