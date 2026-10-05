# Routed FP8 validation on RTX 5070

Measured on 2026-10-05 with RTX 5070 12 GiB, Windows WDDM,
PyTorch `2.13.0.dev20260516+cu130`, NVIDIA driver 595.79.
Source: `F:/AI/models/DeepSeek-V2-Lite-Chat`.
Output: `F:/AI/models/DeepSeek-V2-Lite-Chat-FP8`.

## Checkpoint and execution

All 4,992 routed gate/up/down projections use E4M3FN with scalar FP32
decoding scales (`weight_scale_inv`, shape `[1, 1]`). Backbone, routers,
MLA, norms and shared experts retain their original BF16 bytes. Configuration,
tokenizer and other top-level non-weight files are copied verbatim. The HF
index is rebuilt to include scales; `runtime_fp8.json` identifies this harness
format. This is not a claim of compatibility with arbitrary HF FP8 loaders.

The validated tensor payload is 17,018,137,088 bytes (15.85 GiB). Routed
slots shrink from 17,301,504 to 8,651,264 bytes, including three aligned
scale regions and final 512-byte stride padding. Weight and scale regions
start at offsets divisible by 128. Shared BF16 regions reserve four FP8
slots apiece, with padding kept outside the shared views.

Native `aten::_scaled_mm.out` performs FP8 Tensor Core GEMMs with dynamic
tensorwise FP8 activations, stored weight scales, FP32 output and
`use_fast_accum=False`. Activations after gate/up projections, SwiGLU and
shared experts stay BF16. There is no full BF16 expert-weight expansion.
Persistent activation/output scratch is included in pointer checks. CUDA
tests compare against dequantized operand GEMMs and check expert arena bytes
remain unchanged. The private PyTorch operator requires a compatible CUDA
build; unsupported hardware/builds fail rather than silently falling back.

Transfer scratch decreases from 4 MiB to 2 MiB per tier. Each miss moves
one complete padded slot in each direction. Checkpoint readers are sealed
after complete startup ingestion, with the same exclusive RAM/VRAM residency
invariant as BF16.

## Three-generation comparison

Identical chat-template prompt `Hello`, eight prompt tokens and eight output
tokens. KV visibility resets between runs; expert residency and token clock
persist. Each run executes 15 token steps and 2,340 routed expert requests.
Decode means cover seven steps; prefill is excluded from those means.

| Configuration | Run | Mean decode | Overall hit rate | Swaps | Bytes per PCIe direction |
| --- | --- | --- | --- | --- | --- |
| BF16, 384 slots | 1 | 255.1 ms | 52.0% | 1,124 | 19,446,890,496 |
| BF16, 384 slots | 2 | 256.6 ms | 52.0% | 1,123 | 19,429,588,992 |
| BF16, 384 slots | 3 | 256.0 ms | 52.0% | 1,123 | 19,429,588,992 |
| FP8, auto 971 slots | 1 | 150.8 ms | 76.3% | 554 | 4,792,800,256 |
| FP8, auto 971 slots | 2 | 81.1 ms | 100% | 0 | 0 |
| FP8, auto 971 slots | 3 | 80.8 ms | 100% | 0 | 0 |

Warm FP8 decode is about 3.16x faster than warm BF16 at 384 slots, or
12.3 decode tokens/s. Warm FP8 swap service and both transfer-direction
event durations are exactly zero. Cold FP8 decode averages 59.2 ms in
swap service; warm BF16 averages about 192 ms. The comparison changes
precision and capacity together, so it does not isolate GEMM speedup.

Generation disk bytes, reader requests and source read attempts are zero
for both configurations. File descriptors are closed, resident sets are
disjoint, coverage is 1,664 experts, and arena pointers remain stable.
All six generations produce identical token IDs and text:
`Hello! How can I help you today`.

FP8 uses 971 VRAM slots and 693 pinned RAM slots (5,995,325,952 bytes).
CUDA allocated memory is 11,065,640,448 bytes, including backbone, shared
experts, KV and scratch. BF16 uses 9,310,286,336 allocated CUDA bytes.
Automatic capacity depends on currently free VRAM and KV capacity; it is
not a fixed 971-slot guarantee.

100% retention is demonstrated for this repeated prompt's working set.
The 1,664 FP8 routed slots alone require about 13.41 GiB, so retaining
every expert in a 12-GiB GPU is impossible. New prompts and long conversations
can still trigger RAM/VRAM swaps. Short greedy-output agreement is a smoke
check, not a model-wide quality evaluation.

## CLI and reproduction

The default-context CLI selected 955 slots, initialized cleanly, and produced
the same response: 8 prompt / 8 generated tokens, 2.73 s elapsed,
2.93 output tokens/s including prefill, 75.9% cold hit rate.
This is a cold conversation measurement; the 12.3 tokens/s above is warm
decode-only throughput.

A separate three-turn CLI session, repeating `Hello` while retaining chat
history, measured the following. Its prompts grow because the current chat
harness resets KV visibility and replays history while keeping expert weights
and the token clock alive.

| Turn | Prompt tokens | Output tokens | Wall time | Output tokens/s | Hit rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 8 | 8 | 2.59 s | 3.08 | 75.9% |
| 2 | 25 | 8 | 3.22 s | 2.48 | 94.8% |
| 3 | 42 | 8 | 6.02 s | 1.33 | 88.3% |

Replies remained valid: turns 1/3 produced `Hello! How can I help you today`,
and turn 2 produced `Hello! How can I assist you today`. These CLI wall-time
rates include all prompt replay and cannot be represented as sustained
12.3 tokens/s conversation throughput or universal 100% cache retention.

```powershell
python -m tools.dsv4_moe_runtime.quantize_fp8 --source F:/AI/models/DeepSeek-V2-Lite-Chat --output F:/AI/models/DeepSeek-V2-Lite-Chat-FP8
python -m tools.dsv4_moe_runtime.benchmarks.exclusive_generation --weights-dir F:/AI/models/DeepSeek-V2-Lite-Chat --slots 384 --report bf16.json
python -m tools.dsv4_moe_runtime.benchmarks.exclusive_generation --weights-dir F:/AI/models/DeepSeek-V2-Lite-Chat-FP8 --auto-slots --report fp8.json
python -m tools.dsv4_moe_runtime.chat --weights-dir F:/AI/models/DeepSeek-V2-Lite-Chat-FP8 --auto-slots
```

Evidence: [BF16 steps](fp8_bf16_384.json), [FP8 steps](fp8_auto.json).
Runtime suite: 146 passed, 10 Ascend-only skipped, one existing read-only
mmap warning.

## Interrupted conversion

An initial full-shard writer was interrupted by a Windows reboot. The System
event log records bugcheck `0x1A`, parameter 1 `0x42001`; a dump is recorded
at `C:/Windows/MEMORY.DMP`. This confirms a system memory-management failure;
the driver/hardware root cause has not been established by dump analysis.
Microsoft documents `0x1A` as
[MEMORY_MANAGEMENT](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x1a--memory-management).
The first output shard failed validation (zeroed expert payload) and was
preserved outside the active checkpoint for diagnosis. No failed shard is
referenced by the final index.

The final utility streams one tensor at a time, flushes and fsyncs completed
shards, and validates dtype, shape and exact output bytes against freshly
quantized source tensors. It publishes the index and completion manifest only
after all shards pass. `--resume` revalidates completed shards and continues
an interrupted conversion. Subsequent conversion, benchmarks and tests
completed without another reported crash. This does not establish a repair
of the underlying Windows failure.
