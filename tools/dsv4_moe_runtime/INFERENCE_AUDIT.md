# DeepSeek-V2-Lite real-weight validation

Validated on 2026-10-05 on the user-selected NVIDIA RTX 5070, with PyTorch
2.13.0.dev20260516+cu130 and Transformers 5.8.1. No Ascend execution occurred.

## Checkpoint schema

The loader checks all 5,291 tensor keys, BF16 dtypes, dimensions and byte counts
against the complete expected schema before allocating device weights. The
checkpoint contains 15,706,484,224 parameters (31,412,968,448 payload bytes).

| Component | Verified shape or topology |
| --- | --- |
| Decoder | 27 layers, hidden 2048; layer 0 dense, layers 1-26 MoE |
| Dense MLP | gate/up [10944, 2048], down [2048, 10944] |
| Router | [64, 2048], FP32 softmax, greedy top-6, no renormalization, scale 1 |
| Routed experts | 64 per MoE layer; gate/up [1408, 2048], down [2048, 1408] |
| Shared experts | two combined; gate/up [2816, 2048], down [2048, 2816] |
| MLA query | q_proj [3072, 2048], 16 heads, 128 non-RoPE + 64 RoPE |
| MLA compressed KV | kv_a [576, 2048], RMSNorm [512], kv_b [4096, 512] |
| MLA output | o_proj [2048, 2048], value width 128 per head |
| Vocabulary | embedding and untied LM head [102400, 2048] |

YaRN uses the checkpoint's factor 40, correction range and attention scaling.
The test compares its tables byte-exactly to the checkpoint's independent
rotary implementation, including the interleaved query/key permutation.

## Arena and execution audit

- `StaticExpertSlotPool` reserves 17,301,504 bytes per routed slot and
  34,603,008 bytes per layer's shared region. Shared regions are outside the
  eviction policy. The new decoder loads every shared region before execution.
- Backbone weights occupy a separate fixed BF16 arena. Routed weights stream
  from the four safetensors shards through the existing bounded pinned chunk
  pool into fixed slot views. No full checkpoint copy is placed in host RAM.
- Expanded K/V caches have fixed capacity and addresses. The reference attention
  path allocates intermediates; fixed weights do not establish a whole-decoder
  zero-allocation guarantee. The static plan excludes scratch and rotary tables;
  allocator statistics in the JSON capture their additional residency.
- File offsets need aligned reads and slicing: safetensors payload offsets are
  not universally sector-aligned. Existing ingestion tests exercise the actual
  unbuffered Windows path and compare checkpoint bytes independently.
- Copy streams adopt the producer stream and synchronize before returning and
  recycling pinned chunks. This is safe but serializes each transfer. Routing
  also requires a host transfer of the six selected expert IDs per MoE layer.
- Fixed a launch race in `AclnnV5Backend`: raw aclnn calls now use the caller's
  current stream, rather than creating an unrelated stream on every launch.
- Fixed MoE arithmetic: routed expert results are weighted and summed in FP32,
  cast once to BF16, then added to the shared result. A real-checkpoint regression
  compares against independently read tensors and checkpoint arithmetic.

## Operator boundaries

The old `DraftInferenceEngine` is a harness: synthetic gate scores, uninitialized
embedding storage, dummy expert execution and no LM head. Its aclnn V5 attention
method explicitly raises instead of executing attention. It is not a generator.

The new `inference/dsv2_lite.py` executes embedding, input RMSNorm, MLA projections,
YaRN, causal attention over the populated cache prefix, attention residual,
post-attention RMSNorm, dense or offloaded MoE, residual, final RMSNorm and LM head.
CUDA uses eager Torch operators. The NPU branch calls `torch_npu.npu_rms_norm`
and Torch projection operations on the current stream; it remains unvalidated.

The custom `_C_ascend.dsv4_moe_expert` dispatcher in `csrc/torch_binding.cpp`
calls `aclnnDsv4MoeExpert` through the C++ adapter. Its weights are packed FP4,
with E8M0 block-32 scales, and its SwiGLU clamp is 10. V2-Lite has BF16 weights,
no such scales and no architectural clamp. Reinterpreting V2 bytes as these
operands would compute the wrong model. This custom kernel stays disabled for
V2-Lite; a BF16, unclamped implementation would be needed to enable it.

The custom host/device tiling mirrors three int64 fields and two float fields
(32 bytes): hidden width, intermediate width, block size, clamp and padding.
Gate C copies the structure to device storage before launching its stream; the
registered aclnn path serializes it through the CANN tiling context. Custom
tiling selects one AIV and zero workspace. The V5 Python wrappers use a separate
fixed workspace and reject growth at execution. Their version-dependent ctypes
signatures and plan/launch fallbacks still require verification against the
installed CANN headers on an Ascend host; symbol discovery alone is insufficient.

## Measured generation

Command from the repository root:

```powershell
python -m tools.dsv4_moe_runtime.inference.dsv2_lite --weights-dir F:/AI/models/DeepSeek-V2-Lite-Chat --device cuda:0 --slots 64 --max-new-tokens 8 --prompt Hello --report tools/dsv4_moe_runtime/benchmarks/dsv2_lite_generation.json
```

| Measurement | Result |
| --- | --- |
| Weight loading and initialization | 2.59 seconds |
| Sequential prefill, 8 prompt tokens | 9.24 seconds |
| Mean subsequent decode step, 7 steps | 1.03 seconds |
| Static weight and expanded KV plan | 3.478 GiB |
| Expert arena, 64 routed + 52 shared slots | 1.869 GiB |
| Live device allocation after generation | 3.511 GiB |
| Device allocator reservation | 3.527 GiB |
| Full routed working set on disk | 26.813 GiB |
| Expert loads / hits | 2098 / 242 |
| Streamed bytes, including shared initialization | 37,198,233,600 |

Generated IDs: `[37727, 0, 1724, 481, 304, 1345, 340, 3571]`.
Decoded output: `Hello! How can I help you today`.
All measured logits were finite and weight/cache/scratch pointers stayed fixed.
This is a short greedy smoke test, not an accuracy benchmark or a full-model
logit comparison against Hugging Face. Latencies include disk offload and host
orchestration; sequential prefill is intentionally unoptimized. The run has no
separate warm-up and is not an isolated performance benchmark.

## Verification

`python -m pytest tools/dsv4_moe_runtime/tests/ -q`: 130 passed, 10 skipped
(Ascend-only), with one existing read-only mmap buffer warning. Ruff checks pass
on all six changed Python files. No dependencies were upgraded. No CANN compiler
or NPU hardware is available on this host, so Ascend bindings, device tiling and
custom kernels were inspected in source but not executed.
