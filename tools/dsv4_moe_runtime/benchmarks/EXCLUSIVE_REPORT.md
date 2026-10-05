# Complete exclusive RAM/VRAM hierarchy

Implemented and validated on RTX 5070 using the DeepSeek-V2-Lite-Chat BF16
checkpoint on the Samsung 970 EVO Plus NVMe drive. The decoder and CLI chat
now default to `storage="exclusive"`; the legacy on-demand disk provider is
explicitly selected only by the disk baseline benchmark or `--storage disk`
in the standalone validation command.

## Allocation and invariant

All 1,664 routed experts are ingested at initialization. Available device memory
minus backbone, shared experts, expanded KV cache, transfer scratch and a
512-MiB runtime reserve determines the maximum K. An explicit `--slots` may
choose a smaller K; chat's `--auto-slots` chooses that maximum. The host partition
has exactly N-K expert slots in pinned allocations and a 2-GiB host safety
reserve. Insufficient RAM or VRAM fails startup, without a disk fallback.

| Configuration | VRAM routed slots | Pinned RAM slots | Pinned expert bytes |
| --- | --- | --- | --- |
| Constrained pool | 64 | 1600 | 27,682,406,400 (25.781 GiB) |
| Larger pool | 156 | 1508 | 26,090,668,032 (24.299 GiB) |

The host arena is divided into fixed blocks of at most approximately 1 GiB.
This avoids a failed single large page-lock request and excessive host-allocator
rounding. Every block in both measured runs was confirmed pinned. Backbone,
shared experts and fixed KV storage remain in device memory as before.

At every completed acquisition, the RAM and VRAM expert-key sets are disjoint,
their union contains all N experts, every GPU slot is populated, every host
slot has one owner, and arena addresses remain fixed. Hits perform no weight
transfer. Misses use the existing eviction policy and exchange the incoming
expert's RAM slot with the selected victim's GPU slot.

This is exclusivity of resident ownership, with a necessary bounded transfer
exception: a literal guarantee of no duplicate bytes at any instant during a
copy cannot be provided by CUDA DMA. Swapping overlapping source/destination
slots concurrently would corrupt the weights. One 4-MiB pinned host chunk and
one 4-MiB device chunk preserve the two sources; they are not resident expert
slots or full expert mirrors. Chunks and startup staging buffers are erased
after use. An expert is unavailable for execution while it is being moved.

Each chunk exchange stages the incoming host bytes using CPU memcpy and the
victim bytes using D2D, then issues non-blocking H2D and D2H on separate CUDA
streams. Both copies complete before the scratch is reused. Residency changes
are published after successful exchange and checked before execution resumes.
A partially failed exchange poisons the provider rather than retrying corrupted
weights or falling back to disk. The implementation is single-flight.

## Disk shutdown verification

Initialization reads 28,789,702,656 routed payload bytes once, in addition to
backbone/shared weights. It creates no expert mmap views. All checkpoint reader
descriptors are closed and staging chunk references are released afterward.
Source reads, raw reader reads, buffered tails and new mappings explicitly
raise after shutdown; the buffered-tail path cannot silently reopen its file.

The benchmark deliberately probes two shutdown guards, then replaces all
reader entry points with functions that fail on any generation-time request.
Across all three generations in both configurations:

- Checkpoint disk read bytes: **0**.
- Runtime reader requests and source read attempts: **0**.
- Descriptor-closed, partition-disjoint and stable-pointer checks: passed.
- Generated IDs: `[37727, 0, 1724, 481, 304, 1345, 340, 3571]` in every run.
- Text: `Hello! How can I help you today`, matching the disk-backed reference.

## Three-generation results

Eight prompt tokens plus seven decode steps per generation; the final selected
output token is not forwarded again. All values below are mean decode-step
seconds. Run-1 starts with K experts already loaded during bootstrap, rather
than an empty pool. Runs 2 and 3 reuse the same RAM/VRAM partition.

| Storage / slots | Run-1 step | Run-2 step | Run-3 step | Run-2 swap service | Run-2 GPU H2D | Run-2 GPU D2H |
| --- | --- | --- | --- | --- | --- | --- |
| NVMe baseline / 64 | 1.032 | 1.039 | 1.029 | n/a | included in load | n/a |
| Exclusive / 64 | 0.505 | 0.515 | 0.515 | 0.430 | 0.0796 | 0.0798 |
| Exclusive / 156 | 0.447 | 0.449 | 0.459 | 0.374 | 0.0733 | 0.0735 |

The fresh 64-slot NVMe baseline separately measured **0.762 seconds** per
Run-2 decode step inside blocking file-read calls and **0.931 seconds** in
combined file-read/H2D fill service. The exclusive hierarchy replaces those
reads with approximately **0.430 seconds** of complete exchange service,
reducing mean end-to-end step time by approximately **2x** for the same 64-slot
workload. The historical 156-slot disk baseline ranged from 0.897 to 1.496 seconds
per decode step; it was not remeasured with the new read-only timing counters.

H2D/D2H values are per-direction CUDA-event durations, accumulated across
copies. Their sum is not a full-duplex wall time. Separate streams permit
overlap, but these counters do not prove a particular overlap ratio or peak
PCIe throughput. Synchronized duplex enqueue/wait wall time averages 0.229
seconds per Run-2 decode step at 64 slots and 0.204 seconds at 156 slots. The
remaining swap service includes CPU memcpy, D2D staging, scratch erasure,
Python orchestration and synchronization. These overheads remain an optimization
opportunity; zero disk IO does not mean zero PCIe transfers.

| Configuration | Run | Swaps / hits over 15 forwarded tokens | Bytes each PCIe direction |
| --- | --- | --- | --- |
| 64 | 1 | 2092 / 248 | 36,194,746,368 |
| 64 | 2, 3 | 2073 / 267 | 35,866,017,792 |
| 156 | 1 | 1962 / 378 | 33,945,550,848 |
| 156 | 2, 3 | 1981 / 359 | 34,274,279,424 |

Full partition bootstrap took 14.76 seconds at 64 slots and 14.61 seconds at
156 slots. Live device allocation after generation was 3.515 and 4.997 GiB,
respectively. Pinned transfer scratch is additional to the expert-byte totals.
Short benchmark runs include host orchestration and do not flush SSD caches.

## Reproduction

```powershell
python -m tools.dsv4_moe_runtime.chat --slots 384
python -m tools.dsv4_moe_runtime.chat --auto-slots
python -m tools.dsv4_moe_runtime.benchmarks.exclusive_generation --slots 64 156 --report exclusive.json
python -m tools.dsv4_moe_runtime.benchmarks.cache_generation --weights-dir F:/AI/models/DeepSeek-V2-Lite-Chat --slots 64 --report nvme_baseline.json
```

Evidence: [comparison summary](exclusive_comparison.json),
[64-slot token timings](exclusive_generation.json),
[156-slot token timings](exclusive_generation156.json), and
[fresh NVMe baseline](exclusive_nvme_baseline.json).

Verification: 142 runtime tests passed, 10 Ascend tests skipped. Tests cover
byte-exact repeated swaps on CPU and CUDA, partition coverage, pinned storage,
hit-without-transfer behavior, scratch erasure, fail-stop DMA errors, capacity
planning, closed-reader guards and raw-byte ingestion validation. The existing
read-only mmap warning occurs only in legacy loader tests. This exclusive
swap implementation targets CUDA; its CPU path is for unit tests and no Ascend
full-duplex execution was validated.
