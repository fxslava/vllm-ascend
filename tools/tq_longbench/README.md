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
layer, 16 tokens with the checkpoint's own head counts, is written, prefilled and decoded
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
layers.py        RMSNorm, RoPE, fused projections, the decoder stack, the decode workspace
engine.py        StandaloneModelRunner and StaticDecodeGraph: weights, prefill, the captured decode
hf_bridge.py     give a Hugging Face model's full-attention layers this KV cache
tasks.py         LongBench prompts and metrics, and the synthetic NIAH generator
cpu_reference.py serve the operators from the CPU, for a run with no device
run_eval.py      the evaluation CLI
run_benchmark.py NIAH retrieval, TTFT, decode percentiles and HBM across a context ladder
run_longbench.py the published LongBench tasks across that same ladder
probe.py         per-layer needle mass, entropy, logit lens and divergence from the dense baseline
smoke_dense.py   dense GQA end to end, against eager Hugging Face
smoke_hf.py      a hybrid model through the bridge, against eager Hugging Face
families.py      which family a config.json is: stop tokens, turn markers, prefill policy
glm4.py          GLM-4: config.json to shape, fused-tensor splitting, prefill policy
preflight.py     one synthetic layer through a backend, against exact attention, before any weights
smoke_glm.py     one checkpoint end to end: the folded-o_proj contract, a prompt, NIAH
diagnose.py      per-layer quantisation loss on real weights
```

## The decode step (`--decode-graph`)

A single-token decode of a 3B model is about four hundred kernel launches, and
not one of them runs for as long as it takes the host to submit the next. On an
RTX 5070 that showed up as **42 ms a token of which 42 ms was host time** — the
device idle, waiting for Python. The step is now recorded once and replayed,
which is worth rather more than any arithmetic saved along the way.

Three things had to become true of a step before it could be recorded, and each
is worth having on its own:

**One GEMM where there were three.** `q_proj`/`k_proj`/`v_proj` are one
`qkv_proj` and `gate_proj`/`up_proj` are one `gate_up_proj`. A checkpoint still
names them separately and still loads: `CausalLM.load_destinations` hands the
loader *views* into the fused tensors, so `self_attn.q_proj.weight` is the first
`num_heads · D` rows of `qkv_proj.weight` and copying into it writes the fused
one. A checkpoint that was already fused (`chatglm`'s `query_key_value`,
`dense_h_to_4h`) is split by the adapter and lands back in one tensor in the
order it arrived. cuBLAS splits one GEMM's output rows exactly as it would
compute three, so the fusion is **bit-identical**, not merely close. It also
puts q's heads immediately before k's, which is why RoPE is one call per layer
instead of two.

**Buffers instead of allocations.** `DecodeWorkspace` holds every tensor a
one-token step writes into — the residual stream, the fused projection, the
attention output, the fused gate/up, the logits, and the three indices that aim
them. They are shared by every layer, which is safe because the decode is
sequential: layer *i* has consumed its `qkv` before layer *i+1* writes it.
Nothing on the path calls `torch.empty`.

**No reading the device back.** Every backend's decode used to open with
`int(context_lens.max())` to decide how much of the pool to slice — a
device-to-host copy **per layer per step**, 36 of them a token — and the
tie-point check read the same tensor again. Both are gone from this path. The
decode is handed a `window` (how many slots to read) and the lengths stay on the
device as the mask they already were; slots past a row's own length are masked
off exactly as before, so a window wider than the prefix costs reads and changes
no number. `native_v5` is the one backend this cannot reach: its entry points
take the kv lengths as a Python list, so it reports
`supports_static_decode = False` and is refused a capture rather than given one
whose lengths were frozen at record time.

With those, `StaticDecodeGraph` records the whole step — every layer, the head
and the argmax — into a `CUDAGraph` on CUDA or an `NPUGraph` on NPU, after
`GRAPH_WARMUP_STEPS` (3) real passes. The warmup passes execute, and that is
safe: writing the same token at the same position is idempotent, so a warmup
pass *is* the step, run more than once. Recording executes nothing, so the step
the caller asked for is the replay that follows.

Captures are bucketed by window — powers of two up to the cache — so a 128k run
holds ten graphs rather than 128k of them and no step reads more than twice the
cache it needs. They share one memory pool, because they are replayed one at a
time. The step that captures is timed separately (`captures`, `capture_ms`) and
kept out of the percentiles: it is real time the run spent, and it is not what a
steady-state token costs.

`--decode-graph off` is the same arithmetic over the same buffers, launched a
kernel at a time — the honest thing to compare against, not a different model.
`on` refuses a run that cannot capture, which is what a benchmark wants: a quiet
fallback reporting 26 ms a token reads as slow kernels.

### On an NPU

The step above is written once and takes CANN's own kernels where they exist.
None of this has run on silicon from here — there is no NPU on the host these
numbers were measured on — so each piece is arranged to fail loudly or fall back
rather than to be trusted.

**The operators.** `RMSNorm` calls `torch_npu.npu_rms_norm(x, weight,
epsilon=...)` and unpacks the `(normed, rstd)` it returns; `MLP` calls
`torch_npu.npu_swiglu(fused)`, which halves the trailing axis itself and
computes `silu(first) * second` — exactly what `gate_up_proj` lays out, and what
`vllm_ascend/ops/activation.py` relies on. Both are resolved once, by
`npu_operator`, when the module is built: the decode reaches them a hundred-odd
times a token, and a branch resolved inside a captured region would be recorded
rather than taken. `torch_npu` is never imported to answer — it is looked up in
`sys.modules`, where `resolve_device` put it before torch could parse an `npu`
device at all.

**What a replay must not do.** The audit that mattered for CUDA covers the NPU
paths too, because they share `layers.py` and the `window`. What is left in the
TurboQuant decode is all cached rather than allocated: the rotated-query buffer
and the operator workspace grow on first use and are sized by the 3 warmup
passes before anything is recorded, `context_lens` is already the int32 the
kernel wants, and at one token the query view into the fused projection is
contiguous, so `query.contiguous()` copies nothing. The one allocation left was
`block_table(window).expand(n, -1).contiguous()` — once per layer per step, for a
tensor that depends on nothing but the window — and it is now
`AttentionBackend._block_tables`, keyed on `(tokens, window)` so a captured graph
and the steps after it share one address.

**The memory pool.** Every window's graph shares one pool, and that is safe here
for a reason worth stating rather than assuming: a shared pool lets a later
graph's allocations land on an earlier graph's, so it is sound only when nothing
an earlier replay left in the pool has to survive. Nothing does — the workspace
and both KV pools are allocated in the runner's constructor, outside any pool,
and what is inside is per-step temporaries that are dead when the step ends.
`test_replaying_two_windows_in_turn_keeps_both_correct` is that claim as a test:
capture the small window, capture the large one over the same pool, then replay
the small one again and require the tokens it produced the first time.

**`--weight-nz`.** `npu_format_cast(w, 29)` puts a weight in the Cube's fractal
NZ tiling. `auto` does what `vllm_ascend`'s own `_should_trans_nz` does, which is
not what "native is faster" would suggest: never for float32, always on a 310P,
and for float16 or bfloat16 anywhere else **only at `weight_nz_mode == 2`** —
which the shipped default (`VLLM_ASCEND_ENABLE_NZ=1`, "quantised weights only")
is not. So on a 910 or a 950 the plugin leaves dense half-precision weights in
ND and `auto` does too; `--weight-nz on` is how to ask for the cast and measure
it. Following the vendor here is deliberate: without an NPU to measure on,
copying a considered default beats guessing.

The cast is **checked**. The fused projections of every layer are cast (not
`o_proj`, which carries a fold performed on its ND values), and the first one's
product against a probe vector is compared with the product it formed a moment
earlier in ND. A tiling the matmul reads wrongly does not raise on its own — it
returns numbers, and a model whose first projection is transposed is fluent
nonsense — so a cosine below `NZ_PROBE_MIN_COSINE` fails the load and names
`--weight-nz off` as the way past it. `torch_npu.get_npu_format` confirms the
bytes actually moved, so "cast" does not just mean the call returned.

**What `auto` does when the device disagrees.** `decode_graph='auto'` survives a
capture that fails mid-record: the step it was for is retried eagerly, the run
finishes, and the reason goes on the record instead of showing up as slow
kernels. `on` lets it through, because a benchmark that asked for a graph and
quietly did not get one reports the wrong number. That fallback exists for
exactly the case no machine here can produce — an NPU whose `NPUGraph` will not
record this step.

### What it measured

On CUDA, where it could be measured. Qwen2.5-3B-Instruct, fp16, `cann_dense`,
256-token prompt, 60 decode steps, RTX 5070 (12 GB, ~505 GiB/s measured):

| | p50 | p90 | p99 | tok/s |
| --- | --- | --- | --- | --- |
| before | 51.08 ms | 53.09 | 55.30 | 19.6 |
| fused + buffers + window, eager | 25.59 ms | 27.16 | 28.00 | 39.1 |
| replayed from a captured graph | **12.49 ms** | 12.89 | 13.49 | **80.0** |

Same tokens in all three. The verification run
(`smoke_glm.py --device cuda --backend cann_dense --tokens 256`) reports
p50 12.30 ms / 81.3 tok/s and stops on `<|im_end|>`.

**Where the remaining 12.5 ms goes, and why it is nearly the floor.** BS=1 decode
reads every weight once a token: 6.17 GB for this checkpoint, which at the
~505 GiB/s this card sustains on a large read is **11.0 ms** before any
arithmetic. Timed directly, the four GEMMs over 36 layers plus the head are
10.0 + 1.0 ms of the 12.5, so the step is now **device-bound** — synchronising
around a replay costs 0.08 ms, and seating the three indices and reading the
token back costs 0.04 ms. The other ~1.5 ms is the elementwise work between the
GEMMs (RoPE, the KV write, the attention, the residuals). There is no fp16 path
to 10 ms a token on this card; that needs smaller weights, not fewer launches.

`RMSNorm` calls `torch.nn.functional.rms_norm` where the installed torch has it,
which was the largest single non-GEMM cost: written out it is a promote, a
square, a mean, an add, a reciprocal square root, two multiplies and a cast —
eight launches, seventy-two times a token. The explicit expression is kept as
the fallback and as the statement of what the kernel computes. The two agree bit
for bit on the one-token rows a decode norms; over a 2048-token prefill chunk
they differ on about one element in forty thousand by a single float16 ulp,
which is a summation order rounding differently rather than a different
definition, and changed no token on anything measured here.

## Dumping the cache (`--dump-kv-dir`)

```bash
python tools/tq_longbench/smoke_glm.py --model-path "F:/AI/models/Qwen2.5-3B-Instruct" --device cuda --backend turboquant_reference --tokens 4096 --dump-kv-dir "F:/AI/kvcache"
```

Writes `DIR/<model>_<backend>_<tokens>_<timestamp>/` after the first needle
depth, holding `kv_cache_tensors.pt`, `metadata.json` and a generated
`README.md`. It is for someone analysing the quantised representation offline,
in plain torch, with no `vllm_ascend`, no `torch_npu` and no kernels — so the
dump carries the codec and the rotation, not just the planes.

The cache is trimmed to the blocks the context reached (whole blocks, so the
paging still means what it says) and every tensor is cloned, detached and moved
to the CPU, so a later decode cannot rewrite what was handed over. `metadata.json`
records the architecture, the run's parameters, and the needle's depth, prompt
and whether it was actually retrieved — a cache from a run that lost the needle
is worth analysing, but analysing it as though the model had found it would be
reading the wrong thing, and the tensors carry no hint either way.

**Three things about this cache fail silently**, which is why the dump is a
directory rather than a `.pt` file:

- **It does not hold K.** It holds `Pi @ k`, quantised — the rotation is applied
  before the codec so the 4-bit levels see a flatter distribution. Reconstructing
  without undoing it gives the right shape, the right dtype, and a cosine against
  the true K of **-0.005**. So Pi travels as a `[D, D]` matrix (it is an
  involution, so one matmul undoes it) rather than as a function the consumer
  cannot call.
- **Two codes per byte, low nibble first.** Reading them high-first is also
  silent: cosine **0.000**.
- **The scales are not derivable.** `scale_planes[..., :kv_heads]` are K's,
  `[..., kv_heads:2*kv_heads]` are V's, and the slot is padded to an aligned
  burst beyond that.

The generated README's reconstruction snippet is *executed* by
`test_the_readme_recipe_actually_reconstructs_k_and_v`, against a real cache,
and held to a cosine above 0.99 and below 0.9999 — close because it is correct,
short of 1.0 because it is 4 bits. A guide nobody runs is a guide that drifts,
and every way of getting this wrong produces plausible numbers.

`--dump-kv-dir` needs a backend with a quantised pool and `--tokens` above zero;
both are refused before the checkpoint is read rather than leaving an empty
directory behind.

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

`smoke_glm.py` runs any of these end to end, and reads which one it has from
`config.json`. The shape — head counts, head size, the RoPE base and how much
of each head it rotates — comes from `read_checkpoint_shape` in `engine.py`;
what a config cannot state comes from the *family* `families.py` matches on
`model_type` (below). Nothing in `smoke_glm.py` is one model's.

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

On `transformers` 4.28 GLM-4's tokenizer cannot load as shipped: 4.28 ignores
`tokenizer_config.json`'s `added_tokens_decoder`, so every special-token lookup
raises `KeyError`. `load_tokenizer` suppresses `sanitize_special_tokens` for the
load and registers the declared tokens at their ids — for that family only, since
`Qwen2Tokenizer` has no such problem and shadowing a base-class method to load it
would be a workaround for one it does not have. The result was checked to encode
identically to 4.44. `--dummy-prompt` skips the tokenizer entirely: a 64-token
prompt of ones, short-prompt stage only, for a kernel sanity check.

**Qwen2.5** (`model_type: qwen2`) needs no adapter at all: its `config.json` is
an ordinary HF one, so the shape comes from `ModelShape.from_hf_config` like any
dense GQA checkpoint, and the contrast with GLM-4 is entirely in what the config
says. `Qwen2.5-3B-Instruct` is 36 layers, **16** query heads over **2** KV heads
(8:1 GQA), `D = 128`, RoPE base `1e6` — and NeoX halves over the *whole* head,
where GLM-4 pairs adjacent channels over the first half. It ties `lm_head` to the
embedding and ships no `lm_head.weight`. Having no gate, it folds Π into `o_proj`
on the same contract GLM-4 does.

```bash
python tools/tq_longbench/smoke_glm.py --model-path "F:/AI/models/Qwen2.5-3B-Instruct" --device cuda --backend cann_dense --tokens 256 --max-new-tokens 32
```

`--model-path` is normalised before anything opens it, so either slash, a
trailing separator and `~` all name the same directory; a path that is not a
checkpoint is one sentence rather than a `FileNotFoundError` from inside the
weight load. On a CUDA host `cann_dense` is torch's batched matmul and softmax
over the paged pool and `dense_reference` is `scaled_dot_product_attention` over
the same pool; the TurboQuant kernels need an NPU, and `turboquant_reference`
computes what they compute in torch (correctness only — it says nothing about
latency).

A `model_type` no family claims still runs. It gets `families.GENERIC`, whose
tokenizer's own chat template and config's own `eos_token_id` are the only
authorities on the prompt and the stop set, and which stages no unquantised pool
at any length because what one would cost per token is exactly what is not known
about it. What `GENERIC` never does is lend it another model's ids.

### Stopping, and the prompt format

Two things that used to be decided silently by which `transformers` happened to
be installed, and are now printed before every run:

**Stop tokens.** `glm-4-9b-chat-1m`'s `config.json` declares no `eos_token_id`,
and on the 4.28 path `tokenizer.eos_token_id` is `None` — so the stop set was
**empty** and nothing halted generation. Every continuation ran its whole budget
and was trimmed afterwards, which is indistinguishable from a model with more to
say. Qwen2.5 is the milder form of the same thing: its `config.json` names
`<|im_end|>` and says nothing about `<|endoftext|>`, which its own
`generation_config.json` lists beside it. So each family writes its stop tokens
down **by name** and `eos_ids_for` resolves every one through the tokenizer,
falling back to the published id only if the tokenizer does not know it —
GLM-4's three turn-enders `<|endoftext|>` / `<|user|>` / `<|observation|>`
(151329 / 151336 / 151338), Qwen2.5's `<|im_end|>` / `<|endoftext|>`
(151645 / 151643). `generate(stop_ids=...)` ends the **loop**, not just the
text: a run that generates its budget and is trimmed on the way out has already
paid for every step, so its decode percentiles average over tokens the model
never meant to emit. Which of the two happened is printed after the short
prompt — `stopped on '<|im_end|>' (151645)`, or the budget it ran out against —
because on the page they are the same sentence.

**The chat template.** `encode` takes the first of three routes that works: the
tokenizer's own `apply_chat_template`; the family's turn markers assembled here,
built by id rather than parsed from a string, for a `transformers` that has no
`apply_chat_template` at all; and the bare text. The transcriptions are
`[gMASK]<sop><|user|>\n{prompt}\n<|assistant|>` for GLM-4 and ChatML —
`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n` — for Qwen2.5.
Literal text adjacent to the prompt is tokenised *with* it, in one call, because
a BPE tokenizer merges across the join. Falling through to the third route is not
a neutral default: a chat model handed a document with no turn markers around it
continues the document, which over a needle-in-a-haystack prompt means continuing
the haystack.

Both are reported in stage 1 (`prompt:` and `stop ids:`), and an empty stop set
says so in as many words.

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

## LongBench across the ladder (`run_longbench.py`)

The same rungs, but the published tasks rather than the synthetic needle.

```bash
python tools/tq_longbench/run_longbench.py --model-path /models/glm-4-9b-chat-1m --tasks narrativeqa,qasper,gov_report,multifieldqa_en --contexts 4096,8192,16384,32768,65536 --out-file longbench.jsonl
```

One line per (backend, task, context): the task's own metric (F1, or ROUGE-L for
`gov_report`), TTFT, decode P50/P99, KV megabytes and the allocator's peak.

Context scaling on a fixed dataset **is truncation** — a LongBench item is as
long as it is. Each prompt is cut from the middle, the way the suite itself cuts,
so the head's instructions and the tail's question both survive and a short rung
is the same item with less of its middle. That makes the ladder a measure of how
much of the middle the model needed, which is the question worth asking of a KV
cache. It also means a rung that truncated is **not comparable to published
LongBench numbers**, so the `cut` column counts the items it happened to.

The prompt format and the stop set are `smoke_glm`'s, and both are printed
before the sweep: a task score should not be quietly measuring a missing
`<|assistant|>` marker or a continuation that ran its whole budget.

## Where a long-context answer broke (`--profile-layers`)

`probe.py` reads four numbers off every layer, for the last token of the prefill
— the token whose logits choose the answer's first word:

| column | what it says |
|---|---|
| `needle_mass` | how much of that token's attention landed on the needle's positions |
| `attention_entropy` | mean entropy over heads, in nats, against a `ln(context)` ceiling |
| `target_rank` | logit lens: the layer's state through the *final* norm and the unembedding, and where the expected token ranks |
| `hidden_cosine` | the layer's output against an fp16 `cann_dense` prefill of the same prompt |

The table ends with `L*`, the layer the run stopped making sense at: the first
layer whose hidden state left the dense baseline, or — when no baseline was run
— the steepest fall in needle mass, and it says which of the two it meant.

Only `hidden_cosine` needs two runs. It is also the one that separates "the model
does this anyway" from "the quantised cache did this", which is usually the
question.

**Off costs nothing.** The hooks are registered by the probe's context manager
and removed on the way out, so an unprofiled run has empty hook dicts and torch's
own fast path skips the machinery. `layers.py` does not know the module exists.
On, it costs one extra qkv projection and one attention row per layer per chunk.

## Folding `o_proj` (`--fold-site`)

The fold is `W_o -> W_o (I ⊗ Π)`, and Π is a sign flip and a fast
Walsh-Hadamard transform — `O(D log D)`, not a matmul. Written as a right
multiply by a `[D, D]` matrix it becomes `O(D²)`, 37× the arithmetic, and runs
much faster anyway: one `gemm` is a better thing to ask a machine for than seven
strided float64 passes. At glm-4-9b's geometry (4096 × 4096, 40 layers) on one
6-thread CPU:

| route | 40 layers | vs the shipped transform |
|---|---|---|
| shipped FWHT, float64 | 25.0 s | — |
| matmul, float64 | 1.85 s | 13.5× faster, bit-identical |
| matmul, float32 | 1.23 s | 20.4× faster, within one fp16 ulp |

`--fold-site auto` folds on an accelerator, where the weight is going anyway, and
leaves a CPU run on the shipped transform — not because that is faster there but
because on a CPU it is the harness's own reference. `--fold-site device` takes
the speedup there too. The accumulation dtype follows the device: float64 on a
CPU, float32 on an NPU, which has no fast float64.

float32 is not assumed to be enough. The first layer folded on a device is
checked with the shipped `validate_output_projection_fold` against the weight it
came from, so a device whose float32 matmul is quietly *not* float32 fails the
load rather than folding all 40 layers slightly wrong — a failure with no other
symptom, since the run would be fluent.
