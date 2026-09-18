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
| `native_v5` | `npu_fused_infer_attention_score_v2` (aclnn FIA V5) | dense fp16/bf16 | the baseline; the only path whose numbers are not a quantisation of anything |
| `cann_dense` | `matmul` → `softmax` → `matmul` (aclnnBatchMatMul, aclnnSoftmax) | dense fp16/bf16 | the unquantised baseline with no fused kernel under it — slower, and the one that runs on Ascend 950 |
| `turboquant_cube` | `npu_turboquant_cube_decode` | kv4fp8 packed | one launch: raw-query rotation prologue, attention, un-rotation, `sigmoid(gate)`. float16 only, `block_size % 64 == 0` |
| `turboquant_aiv` | `rotate_q` + `paged_attention` + `rotate_q` | 4-bit packed | every build and both dtypes; the only TurboQuant path with a CPU stand-in |
| `dense_reference` | `scaled_dot_product_attention` | dense fp16/bf16 | the baseline in torch, for a host with no Ascend runtime |
| `turboquant_reference` | torch | 4-bit packed | the 4-bit path in torch: rotate, quantise, pack, dequantise, attend, un-rotate |

`native_v5` and the two TurboQuant backends launch vendor or Ascend C operators
and are **refused** where those cannot run, rather than quietly falling back — a
latency table torch produced under an Ascend backend's name would be worse than
no table. A CPU with the operator stand-ins registered counts as able to run
them; a CUDA device never does.

`cann_dense` is the exception, because it has no such operator to be missing:
`torch.matmul` and `torch.softmax` exist everywhere, and on an NPU they are what
torch_npu dispatches to CANN's `aclnnBatchMatMul` and `aclnnSoftmax`. So it
builds on any device, and is CANN-native only in the sense that on an NPU the
Cube and Vector units do the arithmetic. It materialises the score matrix, which
a fused kernel does not, so a chunk's query rows are tiled to a fixed HBM budget
(`_SCORE_TILE_ELEMENTS`); the tiling changes the launch count, never the result.

### The reference backends

`reference.py` is a functional twin, not a simulation: the arithmetic matches
(packing, Lloyd-Max bins, RMS scale, rotation, the scale plane's burst padding),
the performance does not, and nothing it produces says anything about what a
kernel costs. Its correctness argument is transitive —
`tests/ut/attention/turboquant_cpu_ops.py` is held to the kernels' own goldens,
and the reference is held to *it*:

| golden bound | measured | where |
|---|---|---|
| packed + scale planes byte-identical to the stand-ins | exact | `test_the_packed_planes_are_byte_identical` |
| decode vs stand-ins, 8 heads / 512 ctx | > 0.9999 | `test_the_decode_agrees_at_uniform_and_ragged_contexts` |
| decode vs stand-ins, **16 heads / 2 kv / D=128 / 4096 ctx** | **0.99999995** | `test_the_golden_bound_holds_at_a_real_model_geometry` |
| streaming softmax, window 8192 vs 1365 | 1.00000000 | `test_the_streaming_softmax_does_not_depend_on_the_window` |
| CUDA vs CPU | 1.00000000 | `test_cuda_and_cpu_agree` |

The third row is the geometry of Qwen2.5-3B-Instruct, which is what the CUDA
smoke numbers below rest on.

The Cube decode is the `PRE_ROTATED = false` entry — it takes the raw fp16 query
and changes basis itself, so nothing precedes it and nothing follows it.

> The operator is `torch.ops._C_ascend.npu_turboquant_cube_decode`. There is no
> `torch.ops.vllm.turboquant_fused_decode`; `vllm::turboquant_gated_attention`
> (in `vllm_ascend/ops/turboquant_attention.py`) is the *plugin's* way of routing
> a gate through `unified_attention_with_output`, which this harness does not use.

### The operators without the package build

The Ascend backends need `torch.ops._C_ascend.npu_turboquant_*` registered. The
full extension does that; so does a standalone library holding only the
TurboQuant kernels and their bindings, built in one step:

```bash
python tools/tq_longbench/build_turboquant_ops.py --soc-version Ascend950PR_9599
```

It compiles `csrc/attention/turboquant/standalone/` into
`tools/tq_longbench/lib/libvllm_turboquant_cube.so` and its kernel library. The
flags are `-DVLLM_ENABLE_TURBOQUANT_CUBE=1`, C++17, `-O3` and the given SoC. The
SoC has to be a full variant, because it fixes the core counts. It comes from
`--soc-version`, then `$SOC_VERSION`, then the device: `aclrtGetSocName` via
`ctypes` from the toolkit's `libascendcl.so`, then `torch_npu.npu.get_device_name(0)`.
Either answer is matched against CANN's platform configs. The schemas are the
full extension's own: both include `op_adapter/turboquant_torch_ops.h`.

Only the project's own `vllm_turboquant` install component is installed, so the
output holds the two libraries and nothing else. `ascendc_library()` adds
`asc-devkit` rules that would put a copy under `lib/lib` and launch headers under
`include/`. Leftovers of that kind from an earlier build are removed.

When the operators are missing, the Cube backend (and `ascend_ops()` for the
AIV one) loads the library with `torch.ops.load_library`. It is looked for at
`$TURBOQUANT_LIB_PATH` (a file or a directory; when set, the only candidate),
else in `tools/tq_longbench/lib/`, then in `build/`. Nothing is loaded when the
operators are already registered: a second library would redefine the schemas.
The error names the path and the loader's message when a load fails.

Checked in the vendor image (CANN 9.1.0, torch 2.10, `Ascend950PR_9599`): it
builds with zero warnings, links, and exports every launcher the adapter calls.
Opening it needs the Ascend driver, so the first load happens on an NPU host.

### Ascend 950 and the dense path

Ascend 950 refuses `aclnnFusedInferAttentionScore` V1 to V4 (`EZ9903`), and
op-plugin reaches no further than V4: `torch_npu.npu_fused_infer_attention_score`
launches V2/V3 and `npu_fused_infer_attention_score_v2` launches V4. So on a 950
**every** `native_v5` entry point is refused, whatever the torch_npu build
exports. Elsewhere the three are tried in order — `_v2` first, since that is what
vllm-ascend's own attention calls, then the original entry point, then
`npu_prompt_flash_attention` over the contiguous prefix, which has no block table
and so serves prefill only. Prefill is TND over the paged cache with
`sparse_mode=3` and the compressed 2048 × 2048 int8 causal mask it requires;
decode is one TND row per token with no mask.

Which one a run uses is not guessed. `smoke_glm.py` and `run_benchmark.py` first
run a **pre-flight** (`preflight.py`), before any weights load. One synthetic
layer, 16 tokens with GLM-4's head counts, is written, prefilled and decoded
through every attention path the run will take, and checked against exact
attention in float32. A path that raises fails, and so does one that runs but
returns the wrong numbers.

The fallback chain has two steps, in this order:

1. the first `native_v5` entry point that passes;
2. `cann_dense` — matmul, softmax, matmul — which calls no FIA interface at all
   and is therefore what a 950 settles on. The `dense_staging` pool is built
   through it, and the probes that failed on the way are printed with their
   reasons.

Only when neither runs does the prefill mode change: with `--prefill-mode` left
to default the run prefills through the decode backend (`batched_decode`)
instead, and if `dense_staging` was asked for explicitly it stops with every
probe's reason. `--skip-preflight` bypasses all of this.

Asking for `--backend native_v5` on a 950 is still refused rather than swapped
for `cann_dense`: a named backend is a request to measure that backend. Ask for
`--backend cann_dense` to get the unquantised baseline there.

All five real calls (V5 prefill and decode, V1 prefill and decode, PFA prefill)
bind to torch_npu 2.10.0.post4's schemas and pass its meta kernels at GLM-4's
geometry. Whether a 950 accepts them is what the pre-flight reports.

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
_ascend.py       borrow the shipped layout and rotation, by file path; load the standalone ops library
build_turboquant_ops.py  build that library alone (csrc/attention/turboquant/standalone)
kv_cache.py      StaticKVCache: DenseKVCache and TurboQuantKVCache, allocated once
ops.py           the Ascend backends, their scratch buffers and the tie point
reference.py     the same arithmetic in torch, for CUDA and CPU
layers.py        RMSNorm, RoPE, attention, SwiGLU, the decoder stack, the o_proj fold
engine.py        StandaloneModelRunner: weights, chunked prefill, greedy decode, metrics
hf_bridge.py     give a Hugging Face model's full-attention layers this KV cache
tasks.py         LongBench prompts and metrics, and the synthetic NIAH generator
cpu_reference.py serve the operators from the CPU, for a run with no device
run_eval.py      the evaluation CLI
run_benchmark.py NIAH retrieval, TTFT, decode percentiles and HBM across a context ladder
smoke_dense.py   dense GQA end to end, against eager Hugging Face
smoke_hf.py      a hybrid model through the bridge, against eager Hugging Face
glm4.py          GLM-4: config.json to shape, fused-tensor splitting, prefill policy
preflight.py     one synthetic layer through a backend, against exact attention, before any weights
smoke_glm.py     GLM-4 end to end: the folded-o_proj contract, a prompt, NIAH
diagnose.py      per-layer quantisation loss on real weights
```

## Scoring the needle

`found` is the passcode appearing in the continuation **as a whole token**
(`needle_match`), not as a substring: `894772` does not match `1894772`, so a
model that invented a longer number cannot score. Reciting the needle's sentence
without the number scores nothing.

That verdict is necessary and not sufficient, which is why `needle_report`
travels with it. A continuation can carry the passcode and still be worth
looking at:

| column | meaning |
|---|---|
| `found` | the passcode is in there, wherever |
| `answered_first` | it is the *first* number produced — the model answered rather than wandered into it |
| `first_number` | what it did lead with, so a miss can be read |
| `echoed_prompt` | the needle sentence or the question came back too: recitation, still retrieval |
| `degenerate` | the continuation is a loop |
| `clean` | found, led with it, did not loop |

Both counts are printed, always. `5/5 retrieved, 3/5 clean` is a run that
answered correctly five times and then kept talking twice — which `5/5` on its
own reads as an unqualified pass, and which is exactly how a set of answers like
`894772. The secret passcode for ` and `581850. 581850. 581850. ` came to be
reported as clean. Per-item verdicts are `found`, `found+late`, `found+loop` and
`missed`.

## Tests

```bash
python -m unittest tests.ut._tools.test_tq_longbench -v
```

One GLM-4 attention layer at the lengths the short cases never reach — 32 heads
over 4 kv heads, `D=128`, `rotary_dim=64` interleaved, a 3072-token prefill in
2048-token chunks and a decode over a 4096-token prefix — lives on its own
because it is minutes rather than milliseconds:

```bash
python -m unittest tests.ut._tools.test_attention_layer_cpu_equiv -v
```

Every backend is held to naive causal attention computed in float32 on the host,
never to another backend, and each case prints the cosine it measured rather
than only whether it passed. `TQ_EQUIV_SLOW=0` skips the two long cases and
leaves the RoPE ones; `TQ_EQUIV_DEVICE=npu:0` runs the whole file on a device,
which is the only way to include `turboquant_cube` (it has no CPU stand-in by
design). The cross-chunk case scores the second chunk against a *deliberately
broken* reference as well as the right one, and asserts the two disagree first —
a cosine that only ever sees the right answer says nothing about what it would
do with the wrong one.

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

## Measured baselines

RTX 5070 (12 GB), float16, `reference` backends, 2026-09-18. These are what a
regression shows against.

**Qwen2.5-3B-Instruct** (36 layers, 16 heads / 2 kv, `head_dim` 128) through the
standalone runner — `smoke_dense.py`:

- both paths reproduce **eager Hugging Face 16/16 tokens** on a short prompt, so
  any gap below is the cache's and not the model's;
- **288 B/token/layer** against the dense **1024** (3.56x);
- needle-in-a-haystack at 8192 tokens, five depths: **dense 5/5, TurboQuant 1/5**.

That last number is real 4-bit loss, not a harness fault — the reference is
byte-identical to the operator stand-ins at this exact geometry (table above).
`diagnose.py` localises it, comparing both backends on the *same* query per
layer so error cannot accumulate:

> median **0.9933**, max 0.9968, and two outliers — **layer 0 at 0.8953** and
> **layer 20 at 0.8608**.

Layer 0 is the expected one: its attention is dominated by the sink token, whose
key and value sit far outside the distribution a per-vector RMS scale is chosen
for. Retrieving an exact five-digit code through unshielded layers like that is
where 4 bits runs out.

**Qwen3.5-2B** through `hf_bridge.py` + `smoke_hf.py` (6 of 24 layers claimed):
`cos(dense_reference, unpatched HF) = 1.00000000`, so the bridge is faithful;
`cos(turboquant_reference, unpatched HF) = 0.99532467`, with the argmax
unchanged; 544 B/token/layer against 2048.

**Known limit:** at `max_seq_len=32768`, `dense_staging` allocates both pools
(1.2 GB dense + 0.34 GB packed) beside a 6 GB model and does not fit in 12 GB.
Use `--prefill-mode batched_decode` there, which never allocates the dense pool.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` does not help — it is
unsupported on Windows.

## Model coverage

Two routes, because not every model is dense.

**The standalone runner** (`layers.py` + `engine.py`) builds the model itself and
needs nothing of `transformers` but the tokenizer. It covers **dense GQA with a
SwiGLU MLP** — Qwen2.5 and Qwen3 shapes, including the Qwen3.5
`attn_output_gate` variant (`--attn-output-gate`) where `q_proj` emits
`[query | gate]`. Two things no config states are read off the checkpoint's own
tensor names rather than guessed: `qk_norm` (Qwen3 norms each head of q and k,
Qwen2.5 does not) and `qkv_bias` (Qwen2 carries one with no `attention_bias`
key). Guessing either wrong does not misbehave quietly — it fails to load.

**The bridge** (`hf_bridge.py`) inverts the arrangement for a model the runner
cannot build: `transformers` keeps the weights, the linear attention, the
multimodal RoPE and the vision tower, and only the `full_attention` layers'
attention comes from here. That is the Qwen3.5 case — 18 of its 24 layers are
Gated DeltaNet, which keeps a recurrent state rather than a KV cache, so there is
nothing for TurboQuant to hold in them. It hooks `ALL_ATTENTION_FUNCTIONS`, the
supported extension point, so nothing is monkeypatched.

`--fold-output-rotation` rewrites every `o_proj` to `W_o (I ⊗ Π)` so the decode's
`ROTATED_BASIS` output needs no runtime un-rotation. The fold happens as each
checkpoint tensor is ingested, on the host in float64, and is rounded once into
the run's dtype — never on the device, which has no fast float64. Under
`dense_staging` the unquantised prefill backend rotates its own output by Π too,
because it feeds the same folded projection. A gated layer is refused: the gate
sits between attention and `o_proj`, where the output is still rotated.

**GLM-4** (`THUDM/glm-4-9b-chat-1m`, `glm4.py`) runs on the runner directly. It
is dense GQA with no q/k norm and no gate. Both the `chatglm` checkpoint (fused
`query_key_value` with bias, fused `dense_h_to_4h`) and the HF `glm` export
(split q/k/v, fused `gate_up_proj`) load; fused tensors are split at load time.
The shape is read from `config.json` without running the checkpoint's remote
config code. For the 1M checkpoint that is 32 query heads over **4** KV heads
(`multi_query_group_num`; `glm-4-9b-chat` has 2), `D = 128`, and RoPE base
`10000 · rope_ratio = 1e8`. RoPE rotates the first 64 channels of each head and
pairs *adjacent* channels, which is ChatGLM's `reshape(..., rot_dim // 2, 2)`,
not NeoX halves. Having no gate, it folds Π into `o_proj`, and the Cube decode
runs `ROTATED_BASIS` (`kRotatedBasis`, stage 0): no in-kernel un-rotation.

```bash
python tools/tq_longbench/smoke_glm.py --model-path /models/glm-4-9b-chat-1m --device npu --backend turboquant_cube --tokens 131072
```

Its prefill defaults to `batched_decode` from 32768 context tokens up (and
`run_eval.py` does the same for a GLM-4 checkpoint), `dense_staging` below.
GLM-4-0414 (`model_type: glm4`) adds sandwich norms and is refused.

On `transformers` 4.28 the checkpoint's tokenizer cannot load as shipped: 4.28
ignores `tokenizer_config.json`'s `added_tokens_decoder`, so every special-token
lookup raises `KeyError`. `smoke_glm.py` suppresses `sanitize_special_tokens` for
the load and registers the declared tokens at their ids. The result was checked
to encode identically to 4.44. `--dummy-prompt` skips the tokenizer entirely: a
64-token prompt of ones, short-prompt stage only, for a kernel sanity check.

DeepSeek-V4-Flash is **not implemented** in either route — MLA and the MoE stack
are a separate adapter.

## The context ladder (`run_benchmark.py`)

`smoke_glm.py` answers *does this backend work at all*. `run_benchmark.py`
answers *what does it cost, and does it still retrieve, as the context grows*:
the same needle at every rung of a ladder, with the clock and the allocator read
at each.

```bash
python tools/tq_longbench/run_benchmark.py --model-path /models/glm-4-9b-chat-1m --backends turboquant_cube,cann_dense --contexts 4096,8192,16384,32768,65536 --out-file bench.jsonl
```

One line per (backend, context):

| column | what it is |
|---|---|
| `retrieved` | needles found over `--depths` (default `0, 0.25, 0.5, 0.75, 1`) |
| `ttft ms` | time to first token: the prompt in, the first token chosen — prefill plus the argmax over its logits |
| `p50 us` / `p99 us` | decode latency per token. The first step of a sequence pays first-touch costs no later step does, so it lands in P99 and not in P50 |
| `kv MB` / `dense MB` | what the pools hold, against what the same context would have cost unquantised |
| `peak MB` | the device allocator's own high-water mark, which includes transients no pool budget shows |

A runner's KV pool is static and sized by `max_seq_len`, so **each rung builds
its own runner** — a 4k rung measured inside a 64k runner would report 64k's
memory. That costs a weight load per rung; `--reuse-runner` trades it away for
one runner per backend sized to the longest rung, and the memory columns then
describe that size at every rung.

`--max-new-tokens` defaults to **48**, not to the needle task's own 16: the
answer is a six-digit passcode behind whatever preamble the chat template
invites, and a budget that ends mid-number scores as a miss that was not one. It
is a budget, not a length — generation still stops at EOS. Note the difference
from `smoke_glm.py`, where `--tokens` is context *in*.

Both entry points set `USE_TF=0` and `TF_ENABLE_ONEDNN_OPTS=0` as they are
imported, before anything can reach `transformers`. On a host whose TensorFlow
predates NumPy 2, `transformers`' backend probe is not a slow no-op but an
`AttributeError` out of `np.object` during the import itself.

A record per item is written as it completes (`--out-file`, JSONL), so a ladder
that dies at 64k still leaves 32k and below on disk.
