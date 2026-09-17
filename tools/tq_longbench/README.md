# tq_longbench

A standalone long-context evaluation harness for the TurboQuant 4-bit KV cache.

PyTorch, `torch_npu` and the Ascend C operators; `transformers` for the tokenizer
and the config, `datasets` for LongBench. **No vLLM is imported** — no engine, no
scheduler, no Ray, no HTTP server, no IPC. One sequence at a time, one process.

## Why no vLLM

Two reasons, and only the first is about overhead:

1. A decode step's latency percentiles should measure the kernels, not a
   scheduler.
2. The cache layout has to be the *shipped* one. `tq_longbench/_ascend.py` loads
   `vllm_ascend/attention/turboquant_layout.py` and `turboquant_rotation.py` **by
   file path**, so the harness builds byte-identical codec tables without
   importing the package whose `__init__` reaches vLLM. A second, drifting copy
   of `turboquant_codec_tables` would not raise — it would decode the cache
   against the wrong centroids and return plausible wrong numbers.

`tests/ut/_tools/test_tq_longbench.py` asserts the two routes agree.

## Running it

```bash
python tools/tq_longbench/run_eval.py --model-path /path/to/qwen3.5 --dataset longbench_narrativeqa --backend turboquant_cube --max-seq-len 131072 --out-file results.jsonl
```

`python -m tq_longbench.run_eval` also works, but only with `tools/` **appended**
to `PYTHONPATH`, never prepended: `tools/bisect/` is a package whose name shadows
the standard library's `bisect`, and prepending `tools/` breaks `import random`
— that is, Python itself.

One JSONL record per item, flushed as it completes, so a sweep that dies at 100k
still leaves the 99k that worked. Each record carries the score, the prediction,
prefill throughput, decode P50/P90/P99 in microseconds, and the KV megabytes held
against the dense equivalent.

## Backends (`--backend`)

| | decode | cache | notes |
|---|---|---|---|
| `native_v5` | `npu_fused_infer_attention_score` | dense fp16/bf16 | the baseline; the only path whose numbers are not a quantisation of anything |
| `turboquant_cube` | `npu_turboquant_cube_decode` | kv4fp8 packed | one launch: raw-query rotation prologue, attention, un-rotation, `sigmoid(gate)`. float16 only, `block_size % 64 == 0` |
| `turboquant_aiv` | `rotate_q` + `paged_attention` + `rotate_q` | 4-bit packed | every build and both dtypes; the only TurboQuant path with a CPU stand-in |

The Cube decode is the `PRE_ROTATED = false` entry — it takes the raw fp16 query
and changes basis itself, so nothing precedes it and nothing follows it.

> The operator is `torch.ops._C_ascend.npu_turboquant_cube_decode`. There is no
> `torch.ops.vllm.turboquant_fused_decode`; `vllm::turboquant_gated_attention`
> (in `vllm_ascend/ops/turboquant_attention.py`) is the *plugin's* way of routing
> a gate through `unified_attention_with_output`, which this harness does not use.

## Prefill (`--prefill-mode`)

There is no paged prefill kernel over the 4-bit cache, so the prefix a chunk
attends over has to come from somewhere. Two answers:

- **`dense_staging`** (default at or below 32768 tokens) — the unquantised pool
  is allocated too, prefill attends over it with native FIA (`sparse_mode=3`,
  bottom-right causal), and each chunk is written to *both* pools. Prefill is
  then exactly the reference model's prefill, and the only quantisation in the
  run is the one being measured. Costs a full fp16 KV cache, which is what caps
  it near 32k.
- **`batched_decode`** (default above 32768) — only the quantised pool exists. A
  chunk attends through the TurboQuant decode with one context length per token,
  which is bottom-right causal spelled out row by row. No fp16 pool, so 128k and
  256k fit; the prefix is itself quantised.

`native_v5` requires `dense_staging` and says so rather than decoding out of a
cache it cannot read.

## The tie point

`AttentionBackend.check_tie_point` refuses a decode whose longest
`actualSeqLengthsKv` is not the number of tokens actually written. Both
directions are failures that are otherwise silent: told the context is shorter,
the decode attends over a truncated prefix and returns a fluent wrong answer;
told it is longer, it reads slots that were never written — zeros, which decode
to a centroid rather than to nothing.

## Layout

```text
_ascend.py       borrow the shipped layout and rotation, by file path
kv_cache.py      StaticKVCache: DenseKVCache and TurboQuantKVCache, allocated once
ops.py           the three backends, their scratch buffers and the tie point
layers.py        RMSNorm, RoPE, attention, SwiGLU, the decoder stack, the o_proj fold
engine.py        StandaloneModelRunner: weights, chunked prefill, greedy decode, metrics
tasks.py         LongBench prompts and metrics, and the synthetic NIAH generator
cpu_reference.py serve the operators from the CPU, for a run with no device
run_eval.py      the CLI
```

## Tests

```bash
python -m unittest tests.ut._tools.test_tq_longbench -v
```

Runs on any host — no NPU, no pytest, no vLLM. The CPU operator stand-ins
(`tests/ut/attention/turboquant_cpu_ops.py`) compute what the device computes, so
the suite holds the harness to exact attention rather than to "nothing raised".

Two boundaries it does not cross:

- **The Cube decode has no CPU kernel**, by design. What that launch computes is
  verified on the camodel (`test_sim_950pr_turboquant_fused`). The suite checks
  its host-side half: output-stage selection, and the combinations that are
  refused.
- **Retrieval is tested at the cache, not at a checkpoint.** A needle placed as a
  distinctive K/V pair at a known slot exercises exactly what a 128k NIAH prompt
  would — the write, the block table, the context length and the decode lining up
  across the whole prefix — at 32768 and 131072, at every depth, and with a
  negative control that collapses when the context stops short of the needle.

## Model coverage

Qwen3-shaped dense models, including the Qwen3.5 `attn_output_gate` variant
(`--attn-output-gate`), where `q_proj` emits `[query | gate]` and the gate is
handed to the Cube decode's epilogue rather than applied in torch.

`--fold-output-rotation` rewrites every `o_proj` to `W_o (I ⊗ Π)` so the decode's
`ROTATED_BASIS` output needs no runtime un-rotation. A gated layer is refused:
the gate sits between attention and `o_proj`, where the output is still rotated.

DeepSeek-V4-Flash and GLM-5.2 are **not implemented** — MLA and the MoE stack are
a separate adapter, and `layers.py` currently assumes dense GQA attention with a
SwiGLU MLP.
