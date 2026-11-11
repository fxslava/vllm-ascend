# RFC: Full ACL Graph capture for the TurboQuant decode

Status: proposal. Branch of record: `feat/tq-cube-clean`.
Scope: lifting `ENABLE_TURBOQUANT=1` out of the `PIECEWISE` / `NONE` graph modes
decided in `NPUPlatform._apply_turboquant_graph_mode`.

Nothing in this document has been run on an NPU. There is no `/dev/davinci` on
the development host, so every claim below is either read off the source (cited
by file and line) or named as a thing to measure. Where a claim rests on
torch_npu runtime behaviour rather than on this repository's code, it is marked
**VERIFY**.

---

## 0. Executive summary, and one correction to the premise

The brief names four blockers. Three of them are real and one is already done.

| Brief's blocker | Finding |
| --- | --- |
| 1. Custom op registration / fake-meta kernels | **Already in place and not on the critical path.** `torch.compile` never sees `npu_turboquant_fused_decode`. |
| 2. Host-side synchronisations in the launcher | **Real.** Three capture-illegal transfers and two capture-unsound host reads per launch, per layer. |
| 3. Host-side tiling generation | **Real, and the central design problem.** |
| 4. Stream ordering / allocator lifetime | **Real.** Five device buffers are allocated inside the launch; two of them are ~9 MB each, per layer, per step. |

The correction matters, so it is stated first.

**`splitting_ops` is a piecewise-compilation concern, not the reason full
capture fails.** Under `FULL` and `FULL_DECODE_ONLY` `vllm_ascend` sets
`compilation_config.splitting_ops = []` outright
([`platform.py:1478`](vllm_ascend/platform.py:1478)), so there is no graph break
to remove at all. `FULL_AND_PIECEWISE` does keep the splitting ops, because it
*also* compiles piecewise
([`platform.py:1452-1470`](vllm_ascend/platform.py:1452)) — but that changes
nothing here, because in every one of the three modes attention is reached
through `vllm::unified_attention_with_output`, a vLLM custom op with its own
fake implementation. Dynamo treats it as one opaque node and never traces into
`AscendTurboQuantDryRunBackendImpl.forward_impl`. The TurboQuant op is therefore
invisible to `torch.compile` in every mode, and its C++ `kMeta` kernels
([`torch_bindings.cpp:829-866`](csrc/attention/turboquant/torch_bindings.cpp:829))
exist for `torch.export` and nothing else — the bindings' own comment says so
and it is accurate.

What actually fails is **ACL stream capture**: under `FULL`,
`FULL_DECODE_ONLY` or `FULL_AND_PIECEWISE`, the real C++ launcher runs on a
capturing stream, and it performs host↔device copies, host allocations and host
reads of per-step data on that stream. That is the whole problem. No
`torch.library.custom_op` rewrite, no `register_fake`, and no Dynamo work is
needed for it.

**Second scoping fact, independent of graph capture.** The fused decode reads
the NZ-tiled cache layout; both ingestion writers produce the strip layout; the
two disagree on four independent axes and the only bridge today is
`turboquant_repack_strip_to_nz_tiled`
([`turboquant_nz_ingress.py`](vllm_ascend/attention/turboquant_nz_ingress.py)),
which is deliberately not wired into the serving path. So the work in this RFC
makes a *capturable* decode; it does not by itself make a captured decode that
serves a real model. Those are two separate prerequisites for lifting the
platform gate, and the ladder in §6 keeps them separate.

**Third, and the reason to do this even if full capture never lands.** Every
per-step quantity in the launcher is a function of the *batch*, not of the
*layer*, and is rebuilt and re-uploaded once per layer. On a 60-layer model one
decode step performs ~180 pageable host→device copies and ~300 device
allocations that one copy and five allocations would cover. §4 removes them, and
that change is worth landing on its own under `PIECEWISE`.

---

## 1. Root cause analysis

### 1.1 Where the fallback is decided

[`vllm_ascend/platform.py:1241-1304`](vllm_ascend/platform.py:1241),
`_apply_turboquant_graph_mode`, called from `check_and_update_config`
([`platform.py:547`](vllm_ascend/platform.py:547)):

```python
if not compilation_config.cudagraph_mode.has_full_cudagraphs():
    return
if compilation_config.mode == CompilationMode.VLLM_COMPILE:
    downgraded = CUDAGraphMode.PIECEWISE
else:
    downgraded = CUDAGraphMode.NONE
```

The downgrade is unconditional whenever `ENABLE_TURBOQUANT=1`. It does not ask
whether the kernels are registered, which is the first thing to change: the
function's own docstring justifies the refusal by the *PyTorch golden model*
("it is a PyTorch model of kernels that do not exist yet … the codec is ordinary
torch ops, any one of which may have no NPU kernel and fall back to the CPU"),
and that reasoning applies only when the ops are absent. On an `ascend950`
build they are present — `CMakeLists.txt:204-207` adds
`csrc/attention/turboquant/torch_bindings.cpp` and
`op_host/turboquant_tiling.cpp` to the extension, and `CMakeLists.txt:111-114`
builds all three kernel translation units.

So the gate conflates two paths with different capture properties:

* the **kernel path** (`npu_turboquant_reshape_scatter_aiv` +
  `npu_turboquant_fused_decode`), which is capture-hostile for the five
  mechanical reasons in §1.2 — all fixable;
* the **golden path** (the PyTorch emulation), which is capture-hostile for
  reasons that cannot be fixed (§1.3) and must stay refused.

### 1.2 The kernel path: what breaks capture

All line numbers in
[`csrc/attention/turboquant/torch_bindings.cpp`](csrc/attention/turboquant/torch_bindings.cpp),
inside `npu_turboquant_fused_decode`.

**B1 — `context_lens` is copied across the PCIe boundary on every launch.**
Lines 679-685:

```cpp
const at::Tensor context_lens_host =
    context_lens.device().is_cpu() ? context_lens.contiguous() : context_lens.to(at::kCPU).contiguous();
const at::Tensor context_lens_device = context_lens.device() == query.device()
                                           ? context_lens.contiguous()
                                           : context_lens.to(query.device()).contiguous();
```

Exactly one of the two copies always happens, and the comment says so. The
caller is
[`turboquant_dry_run.py:2665`](vllm_ascend/attention/turboquant_dry_run.py:2665),
which passes `attn_metadata.seq_lens[:batch_size]`. That field is the **CPU
mirror**: `_select_seq_lens`
([`utils.py:594-628`](vllm_ascend/attention/utils.py:594)) prefers
`common_attn_metadata._seq_lens_cpu`, and the builder stores it as both
`seq_lens` and `seq_lens_cpu`
([`attention_v1.py:415-416`](vllm_ascend/attention/attention_v1.py:415)) while
keeping the device copy separately as `seq_lens_gpu`. So the serving case is a
host tensor in, and the H2D on line 683 is the one that runs. On a capturing
stream a synchronous `aclrtMemcpy` fails with 107030, "Stream during the capture
stage is not supported".

The device tensor the op wants already exists and is already a persistent
runner-owned buffer: `attn_metadata.seq_lens_gpu`.

**B2 — the tiling descriptors are built by a host CPU loop over that data.**
Lines 696-699 call
`build_decode_tiling(request, context_lens_host.data_ptr(), nullptr, tiling_host.data_ptr())`
([`op_host/turboquant_tiling.cpp:64-173`](csrc/attention/turboquant/op_host/turboquant_tiling.cpp:64)).
The loop is pure host code, so it is not itself illegal during capture — but its
*result* has to reach the device, which is B3, and its *input* has to be host
visible, which is B1.

**B3 — the tiling buffer is uploaded on every launch.** Line 701:

```cpp
const at::Tensor tiling_device = tiling_host.to(query.device(), /*non_blocking=*/false);
```

`decode_tiling_bytes(tokens, splits, grid)` is `tokens*splits*64 + grid*32`
bytes; at `tokens=256, splits=1, grid≤32` (`kDecodeMaxGrid = 32`,
line 120) that is ~17 KB. Capture-illegal for the same reason as B1.

**B4 — the query basis codec's constant pool is rebuilt on the CPU and uploaded
on every launch.** Lines 747-770. The pool is
`kCodecPoolWordsPerChannel * head_size + kCodecLevels` = 2320 int32 at
`head_size` 256, and the first 1536 words are built by a `std::memcpy` loop over
three butterfly stages. It is a **pure function of `head_size`** — the comment
at line 745 says exactly that ("Built here rather than asked of the caller
because it is a pure function of head_size"). Rebuilding and re-uploading ~9 KB
per layer per step is both capture-illegal and pointless.

**B5 — five device buffers are allocated inside the launch.** Lines 706-709,
747, 794:

| Buffer | Shape | Bytes at `tokens=256, heads=32, splits=1` |
| --- | --- | --- |
| `workspace` (`at::zeros`) | `tokens*heads*splits*(256+kPartialTail)` fp32 | 8.9 MB |
| `query_rot` (`at::empty`) | `tokens*heads*256` fp32 | 8.4 MB |
| `tiling_host` (`at::empty`, CPU) | `decode_tiling_bytes(...)` | 17 KB |
| `basis_pool_host` (`at::zeros`, CPU) | 2320 int32 | 9 KB |
| `mode_tables` (`at::zeros`) | `kMinBurstBytes/4` int32 | 128 B |

`workspace` and `query_rot` are the ones that matter: ~17 MB of allocation plus
an 8.9 MB device zero-fill, per layer, per step.

Whether an allocation *during* capture is legal depends on torch_npu's
`NPUCachingAllocator` supporting a capture-private pool the way CUDA's does.
**VERIFY.** It does not matter much, because this repository already has the
sanctioned answer and it is better regardless: `get_capture_resource`
([`vllm_ascend/compilation/updatable_graph.py:189`](vllm_ascend/compilation/updatable_graph.py:189)),
which outside capture just calls the factory and during capture returns one
resource per key held by the active `UpdatableGraph`. `full_graph_fia` uses it
for exactly this
([`attention_v1.py:665`](vllm_ascend/attention/attention_v1.py:665)).

Note on the user's `record_stream` question: `record_stream` is the wrong tool
here. It governs cross-stream reuse by the caching allocator; a single-stream
capture does not need it. The invariant that matters is **ownership for the
graph's lifetime**: a buffer the graph references must either have been
allocated from the graph's pool during capture (which `UpdatableGraph.capture_end`
assumes — it weakens its resource dict with `weak_ref_tensors`, line 99) or be
owned by something that outlives every replay. A buffer freed back to the
caching allocator between replays and handed to another tensor is the silent
corruption to design against.

**B6 — a runtime capability query per launch.** `aclGetDeviceCapability(0,
ACL_DEVICE_INFO_VECTOR_CORE_NUM, …)` at line 492, in the scatter writer. Not
stream-touching, so probably capture-safe (**VERIFY**), but it is a per-launch
ACL call that belongs in a function-local `static`.

**B7 — two model-static operands are rebuilt in Python on every call.**
[`turboquant_dry_run.py:1628-1629`](vllm_ascend/attention/turboquant_dry_run.py:1628):

```python
rot_tables=_codec_tables(emulator),
h16=_h16_operand(emulator.device),
```

`_codec_tables`
([`:1979`](vllm_ascend/attention/turboquant_dry_run.py:1979)) and
`_h16_operand` ([`:2034`](vllm_ascend/attention/turboquant_dry_run.py:2034))
both go through `_sylvester_hadamard`
([`:392`](vllm_ascend/attention/turboquant_dry_run.py:392)), which builds its
matrix with an `arange`, a bitwise `and`, a `bit_length()`-deep XOR loop, a
`where` and a `to` — roughly eleven device kernels and two allocations, per
layer, per step. `_ensure_statics` already caches `_codec_tables(emulator)` onto
the layer as `layer._turboquant_codec_tables`
([`:2415`](vllm_ascend/attention/turboquant_dry_run.py:2415)) and the decode path
simply does not use it.

**B8 — the batch size, and therefore every slice extent, is a host Python
int.** [`:2653`](vllm_ascend/attention/turboquant_dry_run.py:2653):

```python
batch_size = len(attn_metadata.seq_lens_list)
```

and then `query[:batch_size]`, `block_tables[:batch_size]`,
`seq_lens[:batch_size]`, `output[:batch_size]`. Under capture these extents are
baked in. That is correct-by-accident today only because vLLM pads a captured
batch to the capture size — but `seq_lens_list` carries the *unpadded* request
count plus whatever padding the FIA fix-up added
([`attention_v1.py:387-390`](vllm_ascend/attention/attention_v1.py:387)), so the
captured extent is the first-captured step's, not the replayed step's. This is
**capture-unsound**, not capture-illegal: the replay does not fail, it attends
over the wrong number of rows.

### 1.3 The golden path: what cannot be fixed

These are the reasons `_apply_turboquant_graph_mode`'s refusal is correct *when
the ops are absent*, and they are why the lifted gate must be conditional on the
ops being present rather than unconditional.

* `bool(valid.all())` — a device-to-host synchronisation
  ([`:1375`](vllm_ascend/attention/turboquant_dry_run.py:1375)), followed by
  boolean-mask indexing, whose output shape is data-dependent.
* `torch.unique(pages, return_inverse=True)`
  ([`:1445`](vllm_ascend/attention/turboquant_dry_run.py:1445)) — data-dependent
  output shape, in the value plane's read-modify-write.
* `slot_mapping.tolist()`
  ([`:1126`](vllm_ascend/attention/turboquant_dry_run.py:1126)) — D2H, in the
  prefill strip planner.
* The FIA fallback passes `actual_seq_lengths_kv=attn_metadata.seq_lens_list`, a
  Python list ([`:2682`](vllm_ascend/attention/turboquant_dry_run.py:2682)), so
  the context lengths are baked into the captured launch. This is the specific
  failure the earlier note recorded: *the replays do not fail, they attend over
  a stale length.*
* `_forward_mixed` reads `int(query_lens_cpu.max())` and `.tolist()`
  ([`:2740-2778`](vllm_ascend/attention/turboquant_dry_run.py:2740)). Only
  reachable outside `DecodeOnly`, so irrelevant to `FULL_DECODE_ONLY` and fatal
  to `FULL`.

**Consequence for the design: `FULL_DECODE_ONLY` is the only full mode worth
targeting.** `FULL` captures prefill and mixed batches as well, and
`_forward_mixed` is both host-bound and, by its own docstring, never run on
device.

---

## 2. PyTorch schema and meta registration plan

### 2.1 What is already right

`TORCH_LIBRARY_FRAGMENT(turboquant_ops, ops)`
([`torch_bindings.cpp:880-939`](csrc/attention/turboquant/torch_bindings.cpp:880))
registers all three ops with:

* a schema returning `()` and mutating through `Tensor!` arguments — the correct
  shape for a kernel that writes a caller-provided output;
* a `c10::kPrivateUse1` implementation;
* a `c10::kMeta` implementation, no-op by construction (there is no shape to
  compute), which is what lets a FakeTensor trace or `torch.export` walk the op.

A `torch.library.register_fake` would be strictly redundant with the `kMeta`
registration, and `torch.library.custom_op` cannot express a C++ kernel. **No
change is recommended here.** The one thing the `kMeta` kernel must keep doing
is *not* calling `build_decode_tiling`, since under FakeTensor there is no
storage behind `context_lens` to read — the bindings already say so at line 837
and the meta kernel already complies.

`tests/ut/attention/test_turboquant_op_schemas.py` parses the schema strings out
of the C++ with `torch._C.parse_schema` and checks the Python stubs' parameter
order against them, on a host with no extension. Any schema change below is
gated by that test for free.

### 2.2 The schema changes this RFC needs

Two changes, both additive.

**(a) `tiling` becomes an input of the decode.** The buffer stops being
allocated inside the op and becomes a caller-owned device tensor, so that it can
live for a graph's lifetime at a fixed address.

```
npu_turboquant_fused_decode(
    Tensor query, Tensor pi_signs, Tensor rot_tables, Tensor h16, Tensor? gate,
    Tensor key_cache, Tensor value_cache, Tensor scale_cache,
    Tensor block_tables, Tensor context_lens,
    Tensor tiling,                 # NEW: int32 device, decode_tiling_words(...) elements
    Tensor! workspace,             # NEW: fp32 device, the Split-K partials
    Tensor! query_rot,             # NEW: fp32 device, the rotated query image
    Tensor! output,
    int num_heads, int num_kv_heads, int num_splits,
    int heads_per_task, int output_stage, float scale, float inv_sqrt_len) -> ()
```

`tiling` is read-only to the decode (`Tensor`, not `Tensor!`); `workspace` and
`query_rot` are scratch the kernel writes, so they take `Tensor!`. The op keeps
building `mode_tables` and `basis_pool` itself — see (c) — because neither is a
function of anything the caller knows that the op does not.

The op must then re-check what it used to guarantee by construction: that
`tiling.numel() == decode_tiling_words(tokens, splits, grid)` for the `grid` it
derives, that `workspace.numel()` matches
`tokens*heads*splits*(head_size+kPartialTail)`, and that all three are
int32/fp32, contiguous and burst-aligned. A caller that sizes the tiling buffer
for a different `grid` would otherwise read past it — that is the one new failure
mode this change introduces, and it is a `TORCH_CHECK`, not a hang.

**(b) A new op that fills the tiling buffer.**

```
npu_turboquant_build_decode_tiling(
    Tensor context_lens, Tensor! tiling,
    int num_heads, int num_kv_heads, int head_dim, int block_size,
    int max_blocks, int num_splits, int heads_per_task) -> ()
```

Two implementations behind one schema, which is the point of making it an op:

* **`CompositeExplicitAutograd` / CPU**: `context_lens` on CPU, `tiling` on
  device, host `build_decode_tiling` then one H2D. This is rung 3 of the ladder.
* **`PrivateUse1`**: `context_lens` on device, `tiling` on device, a tiny AIV
  kernel. Rung 5.

The caller does not change between the two. A `kMeta` no-op kernel as above.

**(c) `basis_pool` memoised, not exposed.** Keep it inside the op and give it a
function-local cache keyed by `(head_size, device_index)`:

```cpp
static std::map<std::pair<std::int64_t, c10::DeviceIndex>, at::Tensor> pool_cache;
```

No schema change, no Python change, and the per-launch CPU loop and 9 KB H2D
both disappear. The cache holds device tensors for the process lifetime, which
is the same lifetime the rotation constants already have on the Python side.

### 2.3 Where `grid` comes from

`grid`, `tasks_per_block`, `reduce_tasks_per_block`,
`prologue_vectors_per_block` and `prologue_cube_chunk_vectors` are derived in the
op today ([`torch_bindings.cpp:635-656`](csrc/attention/turboquant/torch_bindings.cpp:635)).
The caller now needs `grid` too, to size the tiling buffer. Do **not**
reimplement that arithmetic in Python — the two sides disagreeing is exactly the
class of bug the host tiling header's own preamble warns about ("nothing in the
toolchain diagnoses two sides disagreeing about which word is which").

Add to `op_host/turboquant_tiling.h`, which is device-free and toolkit-free and
already linked by the bindings, the kernel launcher and the device-free gate in
`tests/ut/ops`:

```cpp
struct DecodeLaunchSplit {
    std::uint32_t grid, tasks_per_block, reduce_tasks_per_block;
    std::uint32_t prologue_vectors_per_block, prologue_cube_chunk_vectors;
};
DecodeLaunchSplit decode_launch_split(const DecodeTilingRequest &r, std::uint32_t max_grid);
```

and expose `decode_tiling_words` to Python through one more trivially-schema'd
op, or — cheaper — have the *builder* op return nothing and have the metadata
builder size its buffer from an upper bound: `max_num_reqs * max_splits * 16 +
kDecodeMaxGrid * 8` words is 1 KB + `max_num_reqs * 64` bytes and is the same
for every capture size. **Recommended**: the upper bound. It removes the
host/device agreement problem entirely at a cost of a few kilobytes, and the op
then checks `tiling.numel() >= decode_tiling_words(...)` rather than `==`.

---

## 3. Zero-host-sync launcher architecture

Three tiers of operand, by what each is a function of. The whole design is this
table.

| Tier | Operands | Function of | Lifetime | Owner |
| --- | --- | --- | --- | --- |
| **0 — model-static** | `pi_signs`, `rot_tables`, `h16`, `basis_pool`, `mode_tables` | `head_size` only | process | `_ensure_statics` (Python) / function-local static (C++) |
| **1 — statically allocated** | the *storage* of `context_lens`, `tiling`, `workspace`, `query_rot` | `max_num_seqs` and the model geometry | one allocation, engine lifetime | `AscendTurboQuantMetadataBuilder.__init__` |
| **2 — per-step** | the *contents* of `context_lens` and `tiling`; `block_tables` | the step | per step, written into Tier-1 storage | the metadata builder |

**Read tier 1 as "statically allocated", never as "static".** The table says
*storage* and *contents* on purpose, and this is the distinction the whole
design turns on:

> **Allocation is init-time work. The fill is per-step work.**

The tiling buffer's size and address are a function of the model configuration
alone, so one maximum-sized allocation at engine start serves every step and
every capture size. Its *contents* are a function of the step's context
lengths — see "Rung 5 is withdrawn" in §6 for the field-by-field citation — so
they are refreshed before every launch, and a graph replay re-reads them from
the same address precisely because nothing moved. An earlier draft of this table
called tier 1 "capture-static", keyed per capture size and owned by
`get_capture_resource`; all four of those were wrong and §4.2 records why.

Everything the launcher did before rungs 1–4 that is capture-illegal is a
Tier-0 or Tier-1 quantity being treated as Tier-2 — a constant rebuilt per
launch, or storage allocated per launch.

### 3.1 Tier 0

* **Python side**: `fused_decode` reads `layer._turboquant_codec_tables` (already
  cached, [`:2415`](vllm_ascend/attention/turboquant_dry_run.py:2415)) instead of
  calling `_codec_tables(emulator)`, and gains an `_h16` cache alongside it —
  either on the emulator or in a module-level dict keyed by
  `(order, device)`. Kills B7.
* **C++ side**: `basis_pool` and `mode_tables` get function-local static caches
  keyed by `(head_size, device_index)`. Kills B4. `aclGetDeviceCapability`
  likewise. Kills B6.

After this tier the launcher performs **zero host→device copies of constants**.

### 3.2 Tier 1

> **Superseded by the implementation.** The `get_capture_resource` keying below
> is not what shipped: the builder allocates one maximum-sized buffer per
> operand in `__init__` and a step takes a `narrow` prefix, so there are no
> capture-size keys at all. §4.2 records why `get_capture_resource` is the wrong
> owner for a host-written buffer and unnecessary for the other two. The
> zero-fill question further down is still open and still worth answering
> (open question 3).

`workspace` and `query_rot` move out of the op into the schema (§2.2(a)) and
were originally to be obtained by the caller through `get_capture_resource`
with a key that includes the capture size, the head count and the head size:

```python
workspace = get_capture_resource(
    (_TQ_WORKSPACE_KEY, num_tokens, self.num_heads, self.head_size, num_splits),
    lambda: torch.zeros(workspace_floats, dtype=torch.float32, device=device),
)
```

Outside capture this is today's behaviour exactly. During capture it is
allocated once and the `UpdatableGraph` holds it (weakly — the graph pool owns
the storage).

One question has to be answered before `at::zeros` becomes a one-time
allocation: **is the zero-fill load-bearing?** `StageTaskOutput` writes a
partial before `TurboQuantPartialReducer` reads it
([`turboquant_vector_service.h:1197`](csrc/attention/turboquant/op_kernel/vector/turboquant_vector_service.h:1197)
and `:290`), so for a live split the buffer is write-before-read and
`at::empty` would do. But a split past the end of a sequence has `num_tiles == 0`
and skips the pipeline without writing its partial, and the reducer's skip
condition for such a split has not been traced. At `num_splits == 1`
(`TURBOQUANT_DECODE_SPLITS = 1`,
[`:684`](vllm_ascend/attention/turboquant_dry_run.py:684)) `NeedsReduction()`
returns false and the question is moot; it becomes live the moment Split-K is
switched on. **Resolve before rung 6.** Conservative answer: keep the
zero-fill, but issue it once per capture rather than once per launch.

### 3.3 Tier 2 — the only per-step data

After tiers 0 and 1, the launch takes exactly three per-step device tensors:
`query`, `block_tables` and `context_lens`, plus the tiling buffer's contents.
`block_tables` is already a device tensor from a persistent runner buffer
(`common_attn_metadata.block_table_tensor`,
[`attention_v1.py:341`](vllm_ascend/attention/attention_v1.py:341)).
`context_lens` becomes `attn_metadata.seq_lens_gpu`
([`attention_v1.py:414`](vllm_ascend/attention/attention_v1.py:414)) instead of
the CPU mirror. Kills B1.

That leaves the tiling buffer's contents, which is the real design question, and
there are three answers.

#### D1 — host-filled, once per step, in the metadata builder (recommended first)

`build_decode_tiling` reads `context_lengths` and launch geometry and nothing
else. For a given model, the geometry is identical for every full-attention
layer, so **the tiling buffer is the same for all 60 layers of a step and is
currently built 60 times.**

Move it to the metadata builder. `AscendTurboQuantMetadataBuilder.build()`
already runs once per step on the host, already holds `seq_lens` as a CPU
tensor, and already performs an H2D for `query_start_loc`
([`attention_v1.py:362`](vllm_ascend/attention/attention_v1.py:362)). It calls
`npu_turboquant_build_decode_tiling` (CPU-dispatch flavour) into the Tier-1
buffer and hands the buffer to every layer on the metadata.

Under graph capture the fill happens **before** the replay, on the replay
stream, writing into the address the graph captured. That is legal and is how
`block_tables` and `seq_lens` already work: the graph reads whatever is at the
pointer, and the step's host work stays outside the capture.

Cost per step: one ~17 KB H2D and one CPU descriptor loop — a 60× reduction on
today's cost, with no kernel change and no device-side arithmetic. This is the
cheapest change that makes the decode capturable.

#### D2 — device-side tiling builder kernel (recommended second)

Replace the host loop with a tiny AIV kernel, one block, reading
`context_lens` from GM and writing `tokens*splits` 64-byte descriptors plus
`grid` 32-byte dispatch records. Then the launch has **no host-visible per-step
input at all** and the H2D disappears.

The arithmetic is cheap enough to be divide-free, which is the crux:

* `num_tiles = (context + 63) >> 6`, `tail_rows = context & 63`,
  `regular_tiles = num_tiles - (tail_rows != 0)` — `kDecodeTileRows` is 64 and
  `kDecodeTileRowsShift` is 6, both already in the header.
* `seq_blocks = (context + block_size - 1) >> log2(block_size)` —
  `build_decode_tiling` already *rejects* a non-power-of-two `block_size`
  (`DecodeTilingStatus::kPageSizeNotPow2`,
  [`turboquant_tiling.cpp:73-74`](csrc/attention/turboquant/op_host/turboquant_tiling.cpp:73))
  and already ships the shift to the device as
  `kTilingWordTilesPerPageShift`.
* `blocks_per_split = CeilDiv(seq_blocks, splits)` — the only genuine divide.
  `splits` is a launch constant, 1 today and 4 or 8 in the roadmap, so
  **constrain `num_splits` to a power of two** and this is a shift as well. If
  that constraint is unacceptable, the host can pass a magic-number reciprocal;
  it should not be a device divide, for the reason the tiling header's preamble
  gives about SPU issue order.
* The dispatch records are a function of launch geometry *only* — no context
  lengths at all
  ([`turboquant_tiling.cpp:142-173`](csrc/attention/turboquant/op_host/turboquant_tiling.cpp:142)).
  ~~**Recommended: a Tier-1 constant** built once per capture size, so the
  dispatch half never changes after warmup.~~ **Withdrawn, and not what
  shipped.** The records' *values* are context-independent, but they live at
  `decode_dispatch_word_base(tokens, splits)`, an offset that moves with the
  token count — so "the dispatch half" is not at a fixed place in the buffer
  and cannot be a constant written once. Separating it would need a second C++
  entry point to save at most 256 words of trivial arithmetic. The whole buffer
  is filled in one call per step instead, which is already the reduction that
  mattered: once per step rather than once per layer per step.

Why a separate launch and not a prologue stage inside the decode: the
descriptors are shared across grid blocks (different kv heads and head chunks of
one token are different tasks, possibly on different blocks, all reading
descriptor `token*splits+split`), so an in-launch prologue would need a
cross-block barrier. `AscendC::SyncAll<false>()` is already reached by a split
launch and already constrains the grid
([`torch_bindings.cpp:642-645`](csrc/attention/turboquant/torch_bindings.cpp:642)),
so it is available — but adding a barrier to the hot kernel to save one
microsecond-scale launch is the wrong trade, and it would invalidate the fused
goldens. A separate kernel keeps the audited decode byte-identical.

The device kernel's oracle is free and exact: **`build_decode_tiling` itself.**
Run both over the same context lengths and compare the buffers word for word.
That is the right remaining use for the host builder, and it is why the builder
should not be deleted when D2 lands.

#### D3 — collapse the tiling buffer into launch scalars (not recommended now)

At `splits == 1` every descriptor word is one scalar load of
`contextLens[token]` plus shifts and masks — cheaper than the six
`tilingGm_.GetValue` scalar loads it would replace
([`turboquant_fused_decode.cpp:300-306`](csrc/attention/turboquant/op_kernel/turboquant_fused_decode.cpp:300)).
`groupHeads_` and `chunksPerGroup_` are read from GM
([`:119-120`](csrc/attention/turboquant/op_kernel/turboquant_fused_decode.cpp:119))
although `numHeads` and `numKvHeads` are already launch scalars, so those two
reads are gratuitous. And the dispatch record is one div/mod chain at block
entry, before the pipeline starts, where the SPU-stall argument does not apply.

So the buffer could go away entirely. **Defer it.** It touches the hot kernel,
the measurement that motivated host tiling has not been repeated for this
variant, and nothing about graph capture requires it. Record it as a
perf follow-up with a named owner, not as part of this work.

### 3.4 The remaining capture-unsoundness: batch size

B8 survives all of the above, because it is in Python and is about extents, not
transfers. Two parts:

1. **The decode must run over the captured extent, not the live one.**
   `_forward_decode` slices everything by `len(attn_metadata.seq_lens_list)`. A
   captured graph must instead cover the full padded extent, and dead rows must
   be harmless. The tiling descriptor already encodes "no work" exactly — the
   all-zero descriptor with `num_tiles == 0`, which both the Cube and the Vector
   role branch on
   ([`turboquant_tiling.h:58-60`](csrc/attention/turboquant/op_host/turboquant_tiling.h:58)).
   So the mechanism exists; what is needed is for a padded row to arrive with
   `context_lens == 0`.

2. **The builder pads `seq_lens` with 1, not 0.**
   [`attention_v1.py:387-390`](vllm_ascend/attention/attention_v1.py:387) pads
   the FIA dummy request with a length of 1 and a block-table row of zeros,
   deliberately, because FIA validates `actualSeqLengthsKv`. A length of 1 makes
   the TurboQuant descriptor *live* with one tile, so a padded row will attend
   over block 0 and write a nonzero output. The comment argues that is harmless
   because the padding tokens' output is trimmed downstream — **that argument has
   to be re-checked for a captured decode, where the padded extent is the
   capture size rather than one dummy request.** This is a decision, not a
   detail, and it is listed in §7.

---

## 4. Pre-allocation strategy and `ModelRunner` warmup

### 4.1 What gets allocated where

> **Superseded by the implementation, and the correction matters.** This section
> originally proposed owning all three Tier-1 buffers through
> `get_capture_resource`. Implementing rungs 1–4 showed that is wrong for the
> tiling buffer and unnecessary for the other two; §4.2 now records why, and the
> layout below is what shipped.

```
engine start
  └─ _ensure_statics(layer, device)            # Tier 0, once per (head_size, device)
        pi_signs, rotation, codec_tables, h16
     op-local statics (C++)                    # Tier 0, once per (head_size, device_index)
        basis_pool, mode_tables, aiv_core_num
     AscendTurboQuantMetadataBuilder.__init__   # Tier 1, once, from max_num_seqs
        context_lens  int32[max_num_reqs]            (device, host-written)
        tiling        int32[upper_bound_words]       (device, host-written)
        workspace     fp32[max*H*S*(D+kPartialTail)] (device, kernel scratch)
        query_rot     fp32[max*H*D]                  (device, kernel scratch)

each step
  └─ builder.build() → _build_backend_metadata  # Tier 2, once per step, all layers
        _refresh_context_lens(...)   host stage + one H2D, FIA padding rows zeroed
        npu_turboquant_build_decode_tiling(context_lens_host, tiling, ...)
  └─ (once the gate lifts) graph.replay()        # no host work inside
```

One maximum-sized allocation serves every capture size, because every size is
exactly computable from `max_num_seqs` and a step uses a prefix (`narrow`, which
keeps the base address). So there are no capture-size keys to get wrong, and no
need for `get_capture_resource`'s `use_max_workspace`
([`updatable_graph.py:101-118`](vllm_ascend/compilation/updatable_graph.py:101))
— which exists for sizes that are *not* exactly computable, and which
`full_graph_fia` needs for that reason.

### 4.2 Why `UpdatableGraph` is *not* needed

Worth stating because it is the obvious thing to reach for. `UpdatableGraph`
exists so an op whose **host scalar arguments** change per step can still be
captured: `register_task` records a handle during capture
([`updatable_graph.py:120-146`](vllm_ascend/compilation/updatable_graph.py:120))
and `GraphUpdateTask.apply` re-issues the launch arguments into the captured node
at replay via `graph_task_update_begin/end`
([`:72-76`](vllm_ascend/compilation/updatable_graph.py:72)). FIA needs it because
`actual_seq_lengths_kv` is a Python list.

TurboQuant does not need it, because after §3 **every per-step quantity is the
*content* of a device buffer at a fixed address, not a launch argument.** The
launch arguments that remain (`num_heads`, `num_kv_heads`, `num_splits`,
`heads_per_task`, `output_stage`, `scale`, `inv_sqrt_len`, and the geometry the
op derives from tensor shapes) are all constant for a capture size.

That is a strictly better position than FIA's and should be preserved
deliberately: *no scalar argument of the fused decode may depend on the step.*
Make it a comment in the binding and a test (§6, G4).

**And `get_capture_resource` turns out not to be needed either.** Open question
2 of §7 is now answered, from the source: `FULL_DECODE_ONLY` capture *does* go
through `UpdatableGraph`
([`model_runner_v1.py:4564-4572`](vllm_ascend/worker/model_runner_v1.py:4564)
builds `ACLGraphWrapper(runtime_mode=CUDAGraphMode.FULL)` for any mode where
`has_full_cudagraphs()`, and that wrapper constructs `UpdatableGraph()`
unconditionally at
[`acl_graph.py:188`](vllm_ascend/compilation/acl_graph.py:188), entering it with
`torch.npu.graph(aclgraph, pool=...)` at
[`:210`](vllm_ascend/compilation/acl_graph.py:210), which calls the override
that sets `_ACTIVE_GRAPH`). So the mechanism is live — and it is still the wrong
owner here, for two reasons the answer does not change:

* **The tiling buffer is written by the host between replays.** A
  `get_capture_resource` buffer is created *during* capture, so its storage
  belongs to the graph's memory pool, and `capture_end` weakens the resource
  dict with `weak_ref_tensors`
  ([`updatable_graph.py:99`](vllm_ascend/compilation/updatable_graph.py:99)) —
  whose own docstring says the point is to keep the *buffer pointer* for the
  graph, not the tensor. That is the right shape for FIA's workspace, which only
  the device touches. It is not a buffer to hand the host for refilling.
* **Outside capture it allocates per call.**
  [`updatable_graph.py:189-194`](vllm_ascend/compilation/updatable_graph.py:189)
  falls through to `factory()` when no graph is capturing, so in PIECEWISE — the
  only mode TurboQuant is served in today — it would keep doing exactly the
  per-step allocation this work removes.

A buffer allocated in the builder's `__init__`, before any capture, is strongly
owned, host-writable, and has one address for the engine's life. That is the
contract `block_tables` and `seq_lens` already have, and it is what shipped.

---

## 5. What the platform gate becomes

Replace the unconditional downgrade in `_apply_turboquant_graph_mode`
([`platform.py:1241`](vllm_ascend/platform.py:1241)) with a conditional one. The
condition is three-part, and all three parts are checkable at
`check_and_update_config` time:

```python
from vllm_ascend.attention.turboquant_dry_run import (
    TURBOQUANT_OP_FUSED_DECODE, TURBOQUANT_OP_RESHAPE_SCATTER,
    _is_op_registered, load_turboquant_extension,
)

load_turboquant_extension()
kernels_present = all(
    _is_op_registered(TURBOQUANT_DEFAULT_NAMESPACE, op)
    for op in (TURBOQUANT_OP_FUSED_DECODE, TURBOQUANT_OP_RESHAPE_SCATTER)
)
```

1. **Both kernels registered.** Without them the golden path runs, and §1.3 is
   unfixable. `_is_op_registered`
   ([`turboquant_dry_run.py:709`](vllm_ascend/attention/turboquant_dry_run.py:709))
   is the existing predicate.
2. **The cache the writers produce is the cache the decode reads.** Until the
   write-time NZ ingress lands, it is not, and a captured decode would read
   bytes in the wrong order. Gate on whatever flag that work introduces; until
   then, hard-coded false.
3. **The mode is `FULL_DECODE_ONLY`.** `FULL` and `FULL_AND_PIECEWISE` capture
   mixed and prefill batches, which go through `_forward_mixed` /
   `_forward_prefill_no_cache` and are host-bound by design. Downgrade those two
   to `FULL_DECODE_ONLY` rather than to `PIECEWISE`, and keep the log line that
   says which and why — it is the one thing in that function that has clearly
   earned its keep.

The declined-not-refused stance in the current docstring stays right: a
configuration that cannot be captured gets a warning and a working mode, not an
exception.

---

## 6. Implementation ladder

Each rung is independently landable and independently gated. Rungs 1–2 are worth
landing under `PIECEWISE` on their own merits; full capture starts at rung 3.

| # | Change | Files | Risk | Gate |
| --- | --- | --- | --- | --- |
| **1** ✅ | Tier-0 statics: cache `basis_pool`, `mode_tables`, `aiv_core_num` in C++; use `layer._turboquant_codec_tables` and a cached `h16` in Python | `torch_bindings.cpp`, `turboquant_dry_run.py` | Low. No schema, no kernel, no layout. | G1, G2, G3 |
| **2** ✅ | `context_lens` becomes `seq_lens_gpu`; the op stops making the host copy; `build_decode_tiling` is fed from a caller-supplied host view | `turboquant_dry_run.py`, `torch_bindings.cpp` | Low–medium. Changes which tensor the op reads; a wrong one is a wrong answer, not an error. | G1, G2, G3, G5 |
| **3** ✅ | Schema change: `tiling`, `workspace`, `query_rot` become inputs; new `npu_turboquant_build_decode_tiling` with the CPU implementation; `decode_launch_split()` in the tiling header | `torch_bindings.cpp`, `op_host/turboquant_tiling.{h,cpp}`, `turboquant_dry_run.py` | Medium. Three new sizing contracts the op must check. | G1, G2, G3, G4, G6 |
| **4** ✅ | Builder ownership: `AscendTurboQuantMetadataBuilder` allocates the Tier-1 buffers itself (see §4.2) and fills the tiling once per step for all layers | `turboquant_dry_run.py` only -- the shared builder is untouched, via its own `_build_backend_metadata` hook | Medium. A new builder class and metadata dataclass. | G1, G3, G6, G7 |
| **5** ❌ | ~~The AIV tiling builder kernel~~ **Withdrawn — see “Rung 5 is withdrawn” below.** Replaced by rung 5′: the capture contract for the host-owned buffer (C1–C6, G8a, G9a, G11a) | `turboquant_dry_run.py` only | Low. No kernel, no schema, no device-side logic. | G1, G3, C1–C5 |
| **6** | Lift the platform gate to conditional `FULL_DECODE_ONLY`; padded-row contract for `context_lens`; refuse capture when the golden path would be taken | `platform.py`, `turboquant_dry_run.py` | High. This is the rung that can corrupt a served model silently. | G1, G3, G10, G11 |
| **7** | *(follow-up, not this work)* D3: remove the tiling buffer; re-measure the SPU argument; `at::empty` for the workspace if §3.2 resolves that way | kernel | — | G8, G9, plus a profile |

### Verification gates

| Gate | What | Where it runs |
| --- | --- | --- |
| **G1** | The three `-Werror` compile tiers still build; the standalone ops build still links | WSL, `build_turboquant_ops.py` in the vendor image |
| **G2** | Schema/stub agreement, parsed out of the C++ | `tests/ut/attention/test_turboquant_op_schemas.py`, no device |
| **G3** | Dispatch-seam behaviour: argument order, the `None` fallback, warn-once | `tests/ut/attention/test_turboquant_dispatch_seam.py`, no device |
| **G4** | **New**: no scalar argument of the fused decode depends on the step. Assert over the schema's `int`/`float` parameters that each is derived from config or geometry, not from `attn_metadata` | new UT, no device |
| **G5** | **New**: the op reads the device `seq_lens`, not the CPU mirror — a recorder stub asserting `context_lens.device.type != "cpu"` | new UT, no device |
| **G6** | **New**: `decode_launch_split()` and the Python-side upper bound agree; `tiling.numel()` checks fire on an undersized buffer | `tests/ut/ops` device-free gate + a UT |
| **G7** | The builder change does not perturb the non-TurboQuant backends | existing `tests/ut/attention` |
| **G8** | **New**: the AIV tiling builder's output is word-for-word identical to `build_decode_tiling` over a grid of context lengths, including zero, a 64-exact multiple, a tail, and a length past `kFusedContextLimit` | CAModel, `S <= 256` |
| **G9** | The fused decode goldens are unchanged: cases (a), (e), (k), (l) under `kv4fp8` | CAModel, `S <= 256`, serial |
| **G10** | **New**: a capture-illegality assertion — run a decode under a faked capture (patch both the forward-context flag and `torch.npu.is_current_stream_capturing`, as `_capture_in_progress()` already does) and assert no `.cpu()`, `.to("cpu")`, `.item()`, `.tolist()` or allocation occurs. A `__torch_function__` mode or a monkeypatched allocator is enough; no device needed | new UT, no device |
| **G11** | End-to-end: `FULL_DECODE_ONLY` and `PIECEWISE` agree token-for-token on a greedy generation. This is the only gate that needs an NPU | 950PR hardware |
| **C1–C5** | **New**: the capture contract for the host-owned buffer — the extent, the address, where the fill runs, the step shapes that are refused, and the staging ring. See rung 5′ below | `tests/ut/attention/test_turboquant_capture_contract.py`, no device |
| **C6** | **New**: the launch reads no host-side metadata field. `TorchFunctionMode` cannot see a Python builtin on a list, which is what C1 turned out to be, so G10 cannot gate this class | same file, no device |
| **G8a** | **New**: `build_decode_tiling` is the only producer of a descriptor, and the host tiling gate now covers *ragged* per-token lengths including zero-length rows | `tests/ut/ops/turboquant/common/decode_tiling_gate.hpp` (g++, no device) + UT |
| **G9a** | **New**: the 21-argument launch ABI is frozen by provenance, so no argument can change position, type or source inside a Python-only change | UT, no device |
| **G11a** | **New**: replay invariance — at one capture extent, steps differing in live content agree on every address, dtype, shape, stride and scalar, and differ only in buffer contents | UT, no device |

G10 is the gate that would have caught every blocker in §1.2 and is the one most
worth writing first — before rung 1, so it fails and then goes green.

### Rung 5 is withdrawn

The original rung 5 moved `build_decode_tiling` onto the AIV so the descriptors
would be produced device-side. **Do not do this**, for three reasons, the first
of which also corrects a tempting misreading of what the tiling buffer is.

**1. The descriptors are not static.** The buffer's *size and address* are a
function of the model configuration alone, which is why one maximum-sized
allocation at engine start serves every step and every capture size (§4.2). Its
*contents* are not. A descriptor's `num_tiles`, `tail_rows`, `regular_tiles`,
`block_start_idx`, `blocks_in_split` and `flags` are all computed from that
token's context length
([`turboquant_tiling.cpp:90`](csrc/attention/turboquant/op_host/turboquant_tiling.cpp:90),
`:94`, `:130`, `:133`), whether a split is live at all is `block_start >=
seq_blocks` (`:118`), and past `kFusedContextLimit` the context length decides
whether the token is split at all (`:93`). The header states it as the
parameter's contract: a worker's tile count is a function of its token's context
length
([`turboquant_tiling.h:235`](csrc/attention/turboquant/op_host/turboquant_tiling.h:235)).

So "build the descriptors once at init" is not an optimisation of this buffer; it
would freeze every worker's span at whatever length init assumed. Left as the
zeros the allocation starts with it is worse than wrong: every descriptor reads
`num_tiles == 0`, both kernel roles skip every worker, and the launch reports
success over an untouched output. Only the *allocation* is init-time work. The
fill is per-step work, and it is already down from once per layer per step to
once per step.

**2. A device-side builder puts an inter-core barrier back on the critical
path.** The descriptors are shared across the grid, so one core would compute
them and the rest would wait — another `AscendC::SyncAll<false>()`, which on
dav-3510 completes only if every launched block is resident at the same time
([`turboquant_tiling.h:223`](csrc/attention/turboquant/op_host/turboquant_tiling.h:223)
records why that caps the grid; the prologue's own barrier is
[`turboquant_query_basis.h:177`](csrc/attention/turboquant/op_kernel/cube/turboquant_query_basis.h:177)).
A second such barrier to save a few hundred bytes of host arithmetic is the
wrong trade in both directions.

**3. It walks back §4.2.** The tiling buffer is host-written *between replays*,
which is why it has to live outside the graph's memory pool. Making it
device-written does not remove the need for a host owner — a device-side builder
still needs this step's context lengths put somewhere — it only adds a producer.

### Rung 5′: the capture contract for the host-owned buffer (landed)

What rung 5 was reaching for is the property that the buffer the host fills
before a replay is the buffer the replayed kernel reads, at the same address and
for the same extent. G10 covers the launch path; these cover the buffer it
launches over.

| Gate | What | Where |
| --- | --- | --- |
| **C1** | The launch extent is the extent the host filled. The builder publishes `turboquant_num_tokens` and the decode slices by it, because the fill side counts rows of `seq_lens` while the op derives `tokens` from `query.size(0)` — and the dispatch records sit at `decode_dispatch_word_base(tokens, splits)`, an offset that *moves with the token count*. A launch whose `tokens` exceeds the filled extent does not trip a size check on a maximum-sized buffer: it reads descriptors out of the dispatch array and dispatch records out of words nobody wrote, and reports success | UT, no device |
| **C2** | All four builder-owned buffers keep one address across steps of different widths and context lengths, and every published view is a prefix at storage offset 0 | UT, no device |
| **C3** | The launch path never calls the tiling builder op. That op takes a CPU tensor, does host arithmetic and ends in an H2D copy: legal in the metadata build, illegal in a capture | UT, no device |
| **C4** | A multi-token decode step (speculative, MTP) is refused rather than mis-tiled. The runner pads tokens to `num_reqs_padded * uniform_decode_query_len` ([`model_runner_v1.py:1095`](vllm_ascend/worker/model_runner_v1.py:1095)), so such a step carries several query rows per request while `seq_lens` carries one length per request, and the descriptor array is indexed by token | UT, no device |
| **C5** | The host staging is pinned and a ring of `TURBOQUANT_STAGING_SLOTS = 2`, and the upload is asynchronous only out of pinned memory. A single pageable buffer copied `non_blocking` is rewritten by the next step under its own in-flight copy — a torn upload, not a stale one | UT, no device |

Each landed with a revert probe: C1 red when the extent falls back to
`len(seq_lens_list)`, C2 red when a published view is cloned, C3 red when the
launch calls the builder op, C4 red when the refusal is removed, C5 red at ring
depth 1 and red when the copy is unconditionally `non_blocking`.

#### C5 is a class, and the repository is full of it

An AST sweep of `vllm_ascend` finds **179** explicit `non_blocking` sites.
Classified by the three C5 questions — is the source pinned, is it a persistent
buffer a later step rewrites, does the flag follow the pinning:

| Verdict | Sites | What it means |
| --- | --- | --- |
| `PERSISTENT?` | 61 | A host buffer reached through an attribute, `non_blocking=True`, no pinning anywhere in the source expression. The C5 shape. |
| `LOCAL-TEMP` | 85 | A temporary built on the spot. Weaker risk: nothing rewrites it, though the copy is still effectively synchronous out of pageable memory. |
| `PINNED-*` | 17 | `.pin_memory()` in the source expression, the common `query_start_loc_cpu.pin_memory().to(..., non_blocking=True)` idiom among them. |
| flag-derived | 16 | The flag is an expression, not a literal. |

The 61 are almost all the engine's own staging — `self.positions_cpu`,
`self.temperature_cpu`, `self.optimistic_seq_lens_cpu`,
`self.input_ids_cpu`, the `_310p` runners' equivalents. **This is the house
pattern, not a TurboQuant defect**, and it is survivable for the same reason it
is pointless: a pageable host-to-device copy is effectively synchronous, because
the runtime stages it through a bounce buffer before returning. So those sites
pay a host wait and gain no asynchrony. Changing 61 of them is neither
justifiable nor verifiable from a host with no device, and it is not in scope
here. They are recorded so that the next person does not read TurboQuant's
pinned ring as gold-plating.

Two sites already do it properly, and one of them is **better than what
TurboQuant ships**:

- `worker/utils.py:255-267` — `self._ids_pinned`, allocated with
  `pin_memory=self.pin_memory`.
- `ops/triton/compressor/compressor_triton.py:394-406` — a pinned staging
  **ring** with a `torch.npu.Event` recorded per slot and `synchronize()`d
  before that slot is reused. That is the rigorous version of C5: it does not
  rest on any argument about stream ordering, where TurboQuant's depth-two ring
  does.

**The assumption TurboQuant's ring rests on, stated.** Depth two is sufficient
only because the staging is written by the host ahead of a single in-order
stream, so by the time the host returns to a slot the whole of the following
step has been enqueued behind that slot's copy. If that stops holding — a
second stream for metadata upload, or out-of-order submission — depth two stops
being sufficient and the fix is the event-per-slot pattern above. That is
deliberately **not** what ships today: it needs a device to exercise, an
`npu.Event` per slot, and a conditional host wait, and an untestable
device-dependent sync path is worse than a documented assumption. C5's gate
records which side is a ring and why (`test_c5_the_staging_ring_has_a_device_slot_per_host_slot`).

The narrow thing that *is* gated: the TurboQuant surface may not pass a literal
`non_blocking` at all, in either direction, so the flag and the pinning can
never disagree.

#### C4: speculative decoding and MTP are refused, deliberately

**Decision: not planned. The refusal is the supported behaviour, not a
placeholder.** It is recorded here because the alternative to a refusal is
silence, and the silence was wrong: nothing in this repository gates TurboQuant
against speculative decoding, so before C4 a user could pass
`--speculative-config` with `ENABLE_TURBOQUANT=1` and get mis-tiled descriptors
with no diagnostic. `_apply_turboquant_graph_mode` (`platform.py:1241`) is the
only TurboQuant-specific configuration check there is, and it is about graph
mode.

**What breaks.** `self.decode_token_per_req = 1 + spec_token_num`
([`model_runner_v1.py:763`](vllm_ascend/worker/model_runner_v1.py:763)), and the
runner pads tokens to `num_reqs_padded * uniform_decode_query_len`
([`:1095`](vllm_ascend/worker/model_runner_v1.py:1095)). So such a step carries
`decode_token_per_req` query rows per request, while `seq_lens` still carries
one KV length per request. The tiling descriptor array is indexed **by token**,
so the extra rows read descriptors the host never wrote.

**Why it is not a Python-only change**, which is the actual reason this is not
being done now:

1. *Per-token context lengths.* `_refresh_context_lens` would expand each
   request's length into one per accepted token — draft slot `i` of request `r`
   attends `seq_lens[r] - (q - 1 - i)` keys for a uniform `q`. That part is
   easy and is Python.
2. *Per-token block tables, and this is the blocker.* `build_decode_tiling`
   reads `block_table + token * max_blocks`
   ([`turboquant_tiling.cpp:123`](csrc/attention/turboquant/op_host/turboquant_tiling.cpp:123)),
   and the decode hands the operator `block_tables[:batch_size]` with
   `batch_size` the token count. Per request and per token coincide only at one
   token per request. With `q > 1` the operator needs either
   - a kernel-visible token-to-request indirection — a new operand, a schema
     change and a kernel change, i.e. exactly the `op_kernel/` work this strand
     has stayed out of; or
   - a host-materialised block table repeated `q` times per step, which is a
     `num_tokens × max_blocks` int32 host-to-device copy **per step** — the
     class of per-step copy rungs 1–4 existed to remove.
3. *Buffer sizing.* The tiling, workspace and `query_rot` buffers are sized
   from `max_num_seqs`; they would need `max_num_seqs * max_q`, which is fine
   but is a real memory increase to justify.
4. *Nothing to verify it against.* G9's goldens are one query row per
   `(token, kv head)`. A spec step has no golden, on the CAModel or anywhere
   else, so the design cannot be checked below hardware.

**If it is wanted, the sign-off question is (2)**: a token-to-request operand on
the op, or a per-step block-table expansion. The first is the right design and
costs a kernel change; the second is Python-only and gives back the per-step
copy. Nothing should be implemented until that is chosen, and neither variant
can be validated without a 950PR.

Until then the refusal names the reason in full, and C4 gates it.

---

## 5a. The platform gate: diagnosis, no verdict

**BLOCKED ON HARDWARE.** This section states what the gate checks, the two
hypotheses, and the one observable that separates them. It deliberately reaches
no conclusion, because the conclusion needs a run on a 950PR.

### What the gate actually checks

`_apply_turboquant_graph_mode` ([`platform.py:1241`](vllm_ascend/platform.py:1241)),
in order:

1. `turboquant_enabled()` — the `ENABLE_TURBOQUANT` flag. Not enabled, nothing
   happens.
2. `compilation_config.cudagraph_mode.has_full_cudagraphs()` — if the mode is
   already PIECEWISE or NONE it returns untouched.
3. Otherwise it overwrites `cudagraph_mode` with `PIECEWISE` when
   `mode == CompilationMode.VLLM_COMPILE` and `NONE` when it is not.
4. It logs the warning.

**The first finding is about the log line itself.** The message a user sees —
*"The TurboQuant dry run cannot be captured under FULL_AND_PIECEWISE … Falling
back to PIECEWISE"* — is emitted by this function, at configuration time,
before anything is captured. **No capture was attempted and no ACL call failed.**
It is our own policy decision, not a platform refusal, and it probes nothing:
not the part, not whether the kernels are registered, not the model. Any full
mode plus `ENABLE_TURBOQUANT=1` produces it.

So the question is not "why did capture fail" but "would it", and that has never
been tested.

### The two hypotheses

**H1 — our code.** The fused decode launch is capture-legal after rungs 1–4 and
the C1–C6 contract, and everything still uncapturable is ours:

* the golden fallback paths the backend keeps (§1.3) — a device-to-host sync,
  two data-dependent shapes, and an FIA call whose context lengths are a Python
  list;
* the ingestion's host-side reads of `seq_lens_list` to size its own work;
* any codec op with no NPU kernel, which falls back to the CPU, which is a D2H,
  which is 107030. The gate's docstring is explicit that this is a property a
  PyTorch emulation cannot promise.

Under H1 the gate is over-broad and the fix is to narrow it: refuse capture when
a golden path would be taken (§5's three conditions), not whenever TurboQuant is
on.

**H2 — platform.** Even with every host operation removed, the fused launch
itself cannot be captured: the MIX launch configuration, the in-launch
`aclGetDeviceCapability` (B6, now a function-local static but still an ACL call
inside a launch), or `AscendC::SyncAll<false>()` meeting a capturing stream.
Under H2 the gate stays, and it should then say *that* rather than what it says
now.

Note which way the evidence currently leans and why it is not enough: the
decode's Python half is measurably clean (G10: 8 torch calls, 3 distinct, no
sync; C6: no host-metadata read), and the C++ half allocates nothing and copies
nothing by schema. That makes H1 *plausible* for the decode launch in isolation,
and says nothing at all about H2's three device-side candidates, none of which
is observable from here.

### The distinguishing criterion

One run of `tools/tq_capture_probe.py`, which captures **only**
`_forward_decode` — cache already written, kernels registered, no model around
it, so a refusal is attributable to the launch rather than to any layer or
fallback. Then:

| Observation | Reads as | Why |
| --- | --- | --- |
| Capture succeeds, replay bit-identical to a fresh eager step | **H1** | The launch is capturable. The gate is policy; narrowing it is rung 6, and G11 proper is still what lifts it. |
| ACL **107030** naming a **memory copy** | **H1** | A host-to-device copy survives on the path. The Python half is gated, so it is in the binding or the ingestion. |
| ACL **107030** naming the **kernel launch** | **H2** | The launch configuration or a sync primitive is refused on a capturing stream. Re-run with `ASCEND_GLOBAL_LOG_LEVEL=0` to see which stream operation the runtime rejected. |
| Any code **other than 107030** | **H2** | A different refusal entirely; the code verbatim is the evidence. |
| Capture and replay both succeed, output **wrong** | **Neither — do not lift** | The worst outcome and the reason the gate is conservative. C2 covers addresses and G11a shapes and scalars, so a mismatch is a per-step quantity neither models, and it becomes a new case in them. |

**The criterion in one line: the ACL error code, and the stream operation it
names.** 107030 on a copy is ours. 107030 on the launch, or any other code, is
the platform's.

### What the probe needs that this host cannot provide

The probe stops at stage 1 without `tools/tq_capture_fixture.py`, which must
supply `build_decode_fixture(torch, tq, args) -> (impl, metadata, query, output)`
over a **written** three-plane cache. That is on purpose: a probe over an
unwritten cache captures the refusal path (`scale_cache is None`), not the
launch, and would answer a different question while looking like it answered
this one. The shortest route to that fixture is whichever harness has already
ingested a model on the target host.

### Explicitly not concluded

Whether the gate reflects a platform limitation or a bug in our code. It needs
the run. Until then `platform.py` is untouched and the gate stays closed.

---

## 5b. Blocked on hardware — exactly what to run

Nothing in this list has been run. Each entry names the gate, the precondition,
the command, and what its outcome decides.

### 1. The capture probe — the H1/H2 criterion (§5a)

Needs: a 950PR, a wheel built with `SOC_VERSION=ascend950` so
`torch.ops.turboquant_ops.npu_turboquant_fused_decode` is registered, and
`tools/tq_capture_fixture.py` supplying `build_decode_fixture` over an already
written three-plane cache.

```bash
python tools/tq_capture_probe.py --tokens 4 --context 256
# if stage 2 fails and the message is ambiguous about what 107030 names:
ASCEND_GLOBAL_LOG_LEVEL=0 python tools/tq_capture_probe.py --tokens 4 --context 256
```

Decides: whether the `platform.py` gate is over-broad policy (H1) or a platform
limitation (H2). The verdict table is in §5a. Exit 0 reads as H1; exit 1 means a
verdict is needed from the error text; exit 2 means a precondition was missing
and **nothing was measured**.

### 2. G9 — the fused decode goldens on the CAModel

Needs: the `csrc/tests/` CAModel tree, which **is not on this branch**. It lives
on `feat/tq-v0.26-clean`, `feat/tq-sink-lse-fusion`,
`feat/ascend-950pr-turbo-quant-scaffold` and `last-stable-point` as
`csrc/tests/sim/test_sim_950pr_turboquant_fused.cpp`. So G9 is not runnable from
`feat/tq-cube-clean` at all without merging or cherry-picking that tree first —
state which, because the answer is part of the result.

Then, inside the `ascend-c-dev` container at `/workspace`, serial and
unbuffered, `S <= 256`, cases (a), (e), (k), (l) under `kv4fp8` only:

```bash
stdbuf -o0 -e0 ./test_sim_950pr_turboquant_fused --gtest_filter='*kv4fp8*'
```

Decides: whether this change set moved the kernel's answer. G9a already freezes
the launch ABI device-free, so a G9 failure would mean something outside the
argument tuple changed — which would be surprising and worth stopping for.

### 3. G11 — `FULL_DECODE_ONLY` against `PIECEWISE`, token for token

Needs: a 950PR, a served model, and the `platform.py` gate bypassed. There is
**no** escape hatch for that today, by design, so this needs a local one-line
change to `_apply_turboquant_graph_mode` — not a committed one, and not before
the probe in (1) has reported.

```bash
# baseline, the mode that is served today
ENABLE_TURBOQUANT=1 python -m vllm.entrypoints.openai.api_server \
    --model <model> --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"PIECEWISE"}' \
    --seed 0 ...
# then the same prompts under the mode under test, with the gate locally bypassed
ENABLE_TURBOQUANT=1 python -m vllm.entrypoints.openai.api_server \
    --model <model> --max-num-seqs 8 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --seed 0 ...
```

Compare greedy generations token for token. Decides whether the gate may be
lifted. **This is the only gate that may lift it**; G1–G10 and C1–C6 together do
not, for the reason at the top of the risk list.

### 4. G8 proper — retired

There is no AIV tiling builder to compare against (rung 5 is withdrawn), so G8
has no hardware leg left. Its invariant is now device-free and already runs:

```bash
python -m pytest -p no:xonsh --noconftest tests/ut/ops/test_turboquant_contract_gates.py -q
```

That builds `test_contract_gates.cpp` and `test_layout_bridge.cpp` with
`-Wall -Wextra -Werror` and no CANN, and `test_contract_gates` drives the host
tiling gate's uniform **and ragged** grids. Verified green, and verified red
under both revert probes, through this driver.

### 5. Not blocked but not run here either

`G1`'s standalone-ops build needs the pinned vendor image
(`quay.io/ascend/vllm-ascend:v0.26.0rc1-a5`); it builds there but cannot load
without a driver, so it is a compile gate only. No C++ under `csrc/` changed in
this strand, so there is nothing new for it to catch — the only C++ touched is
`tests/ut/ops/turboquant/common/decode_tiling_gate.hpp`, which is header-only,
needs no wheel or `ascendc_library` rebuild, and is covered by (4).

### Risks, ranked

1. **Silent wrongness at rung 6.** A captured decode that reads a stale buffer
   or the wrong extent produces plausible tokens. Nothing short of G11 catches
   it. Do not lift the gate on the strength of G1–G10.
2. ~~**The padded-row contract (§3.4).**~~ **Resolved.** The FIA dummy row's
   length of 1 made a dead row live; `turboquant_context_lens` publishes zeros
   past `num_reqs` while `seq_lens`/`seq_lens_list` keep their ones. Gated by
   the builder's padded-row test, and the zero-length descriptor is now covered
   in C++ by the host tiling gate's ragged grid.
3. ~~**Tier-1 ownership (§4.2).**~~ **Resolved, and not via
   `get_capture_resource`.** The buffers are allocated in the builder's
   `__init__`, before any capture, so the question of whether capture reaches
   `UpdatableGraph` stopped mattering for ownership. Address stability is gated
   by C2 and replay invariance by G11a.
4. **The workspace zero-fill (§3.2).** Only bites once Split-K is on, which is
   the roadmap.
5. ~~**`num_splits` constrained to a power of two at rung 5.**~~ Moot: rung 5 is
   withdrawn, so nothing needs a device-side shift.
6. ~~**`ascendc_library` has no incremental rebuild**, so rung 5 iterates at
   clean-build speed.~~ Moot for the same reason; it still applies to rung 7.
7. **A multi-token decode step is refused, not served.** C4 makes that explicit
   instead of silent, but it is a real limitation: speculative decoding and MTP
   need the per-request context lengths expanded per accepted token before this
   decode can tile such a step.

---

## 7. Open questions

These need an answer from outside the code.

1. ~~**Padded rows.**~~ **Decided and implemented: zero.**
   `AscendTurboQuantMetadataBuilder._refresh_context_lens` publishes
   `turboquant_context_lens` with the FIA padding rows at 0 while
   `seq_lens`/`seq_lens_list` keep their ones. A 1 makes
   `build_decode_tiling` write a *live* descriptor, so the padded row attended
   over page 0 and wrote a nonzero result -- which was happening under
   PIECEWISE too, not only under capture. (§3.4)
2. ~~**Does `FULL_DECODE_ONLY` capture in this repository go through
   `UpdatableGraph`?**~~ **Answered: yes** (`model_runner_v1.py:4564` →
   `acl_graph.py:188`). The buffers are owned by the metadata builder anyway;
   §4.2 records why `get_capture_resource` is the wrong owner for a
   host-written buffer and unnecessary for the other two.
3. **Is the Split-K workspace's zero-init load-bearing for a split with
   `num_tiles == 0`?** Trace `TurboQuantPartialReducer`'s skip condition. (§3.2)
4. **Is `num_splits` ever not a power of two?** If it can be, rung 5 needs a
   magic-number reciprocal instead of a shift. (§3.3 D2)
5. **Does the write-time NZ ingress work have a flag yet?** Gate part 2 in §5
   needs one. Until then the lifted gate stays closed for serving even when the
   plumbing is done.
6. **Is `aclGetDeviceCapability` legal on a capturing stream?** It should not
   touch the stream at all, but it is an ACL call inside a launch. (§1.2 B6)

---

## 8. Appendix: the per-step, per-layer cost being removed

Measured by reading the source, not by profiling — there is no msprof baseline
for this path and that remains the gap. At `tokens=256`, `heads=32`,
`num_kv_heads=4`, `head_size=256`, `splits=1`, 60 full-attention layers, one
decode step:

| Quantity | Today | After §3 |
| --- | --- | --- |
| pageable H2D copies | 180 (`context_lens`, `tiling`, `basis_pool`, ×60) | 1 (rung 4) → 0 (rung 5) |
| host CPU descriptor loops | 60 | 1 → 0 |
| CPU butterfly-table loops (1536 words) | 60 | 0 |
| device allocations | 300 | 0 (Tier 1 pre-allocated) |
| device bytes allocated | ~1.0 GB | 0 |
| device zero-fill bytes | ~534 MB (`workspace`) | 0 per step |
| incidental device kernels (`_sylvester_hadamard` ×2) | ~660 | 0 |
| `aclGetDeviceCapability` calls | 60 | 1 per process |

The 1.0 GB figure is allocation churn through the caching allocator, not
footprint — but the zero-fill is real bandwidth, and ~534 MB of HBM writes per
decode step is worth removing whether or not full capture ever lands.
