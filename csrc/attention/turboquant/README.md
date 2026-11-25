# TurboQuant kernels and Torch bindings

The standalone tree at `turboquant-core/src/turboquant` shares the layout and
source bytes below. `SOURCE_SYNC.json` records both baseline revisions and the
current hashes. The standalone telemetry parser and the plugin bindings are
listed as repository-specific sources.

| Path | Contents |
| --- | --- |
| `common/` | Launch and packet ABIs, tiling, workspace contracts, cache addressing and codec constants |
| `ingress/native/` | BF16 prefill batch and per-token scatter writers for the native packed cache |
| `decode/native/` | Wave-v11 native BF16 pre-rotated and raw-query decode, with Cube, Vector and reduction routines |
| `decode/e8/` | Prepared-packet E8 reader and parallel BF16/INT8 dequantization |
| `bindings/` | Direct ASC E8 entry adapters and Torch ingress, decode and registration translation units |

Run `python scripts/check_turboquant_sync.py --core <turboquant-core path>`
from the plugin repository to check inventories, exact source bytes and
comment removal. The standalone build declares legacy launcher prototypes by
defining `TURBOQUANT_LEGACY_CANN_LAUNCH`. The plugin's ASC build uses
host-visible entry definitions. Both keep the same public launcher signatures.

### What the current port changed

The Wave-1 integration established these launch and layout contracts.

* **The prefill is mode-2 only.** There is no `template <bool Mode2>`, no
  `..._mode4_single_shot` entry point and no `reuse_mode4_flags` launcher
  argument. Mode 2 lowers to `ffts_cross_core_sync`: each id is a counter in
  0..15 that banks unconsumed announcements, so recycling one across pipeline
  generations is sound by construction and needs no single-shot allocation to
  bisect against. The three channels are `kSync2PanelReady = 8`,
  `kSync2PanelFree = 9`, `kSync2PageDrained = 10`, single-sourced in
  `common/turboquant_wave_protocol.hpp`, each behind its
  `HardEvent::x_S` drain. `bool reuse_mode4_flags` is therefore out of the
  `npu_turboquant_reshape_and_cache_a8` schema.
* **The prefill pipeline is seamless across strip boundaries.** The outer strip
  loop is gone. A block owns `tokens_per_core` tokens = `tokens_per_core / 16`
  strips and runs them as one chunk stream of `strips * kChunks` columns; column
  *j* stages chunk *j* and multiplies chunk *j − kIngressLookahead*, and both
  indices cross strip boundaries, so MTE2 does not drain at the end of a
  16-token strip. The ring periods all divide `kChunks` (static_asserted), so a
  boundary never shifts which slot a chunk owns. The binding still passes
  six-strip `tokens_per_core = 96` when the batch is a multiple of 96 tokens,
  and `16` otherwise. This admits the frozen Slot03 metadata-hoist schedule
  without dropping a partial share or allocating staging buffers.
* **The fused decode is bf16 only.** The fp16/`half` template instantiations are
  gone, and the dispatch `return`s without launching for any other
  `AscendType`. `torch_bindings.cpp` used to map any non-bf16 query to
  `AscendType::FP16`, which over this kernel would leave `output` whatever
  `at::empty` gave it and report success; it now `TORCH_CHECK`s bf16. The fp16
  chain was already dead in the wheel, so this is a refusal rather than a
  regression.
* **The scatter writer's `launch_probe` is gone** — the parameter, its UB arena
  slot and its publication block, from the kernel, the `__global__` entry and
  the launcher. It was a synthetic buffer that published `GetBlockIdx` /
  `GetBlockNum` / `GetSubBlockNum` back to a host runner, and nothing here ever
  read it.
* **The host tiling is a translation unit.** `build_decode_tiling` was an
  `inline` function in a device-side header, so every consumer of the geometry
  pulled in the kernel tree; it is ordinary host C++ in
  `common/turboquant_tiling.cpp` now, and the binding, the kernel launcher and
  the device-free gate all link the same object. `include/turboquant_fused_decode.hpp`
  is gone, superseded by it.

## Wave-1 / Wave-3 integration (2026-11-25)

The native decoder uses the verified Wave-v11 paired schedule and register
softmax. Its telemetry closure is restored together with the shared UB and
GEMM-reducer budgets; it reuses dead query/workspace storage. The batch writer
uses only frozen Slot03's scalar metadata hoist. Existing wide cache offsets
are preserved. Tensor API names match the nondeprecated CANN 9.2.1 API.

The E8 sources under `decode/e8/` derive from the sealed Wave-3 decoder.
The plugin's `bindings/asc_e8_reader.cpp` and
`bindings/asc_e8_component.cpp` provide host-visible ASC entry definitions.
Independent frozen CPU reference fixtures live under `tests/host/turboquant/e8/`.

The separate `turboquant_e8_ops.reader` and `.dequant` registrations use
caller-owned buffers. `vllm_ascend.attention.turboquant_e8` supplies matching
Python calls and CPU packet/offset preparation helpers. Load the extension
before replay using the existing `load_turboquant_extension()` bootstrap.
Output/witness buffers must be disjoint. Runtime wrappers allocate no tensors.

E8 planes are UINT8 `[packets, 4352]`: bytes 0..4095 hold 2048 little-endian
UINT16 labels (16 bits per eight coordinates); bytes 4096..4351 hold 64 FP32
scales. The K stride of 1056 bytes is internal UB padding. Offset tables use
INT64 byte offsets prepared from validated packet IDs, padded to four entries
for each 32-byte DMA burst. They are distinct from native INT32 block tables
and slot mappings, signed-nibble INT8 planes and separate FP32 scale planes.

The reader supports one KV task, six FP32 query heads in cache basis padded
to `[16, 256]` with rows 6..15 zero, an even 4..256 tiles, and a 1..64-row
tail. BF16 output remains in cache basis. K/V BF16 witnesses and raw FP32 score
witnesses remain mandatory, preallocated parts of the verified ABI. Raw-query
composition and automatic E8 selection in the native attention backend remain
gated by the unverified encoder. No Wave-2/Wave-3 reshape mutant or encoder is
promoted. The measured ticks are historical CAModel results, not new hardware
measurements or a speedup guarantee for this integration.

See `reports/wave_champions_20261125/REPORT.md` at the repository root for source
seals, ABI verification, exact host commands and the remaining dependency and
hardware gates. No full device plugin build or device execution was attempted.

`common/turboquant_layout.h` is mirrored **in full** now, including the
four ingress helpers `NzTiledChannelByte`, `NzTiledChannelNibble`,
`NzScaleSlotFloats` and `NzTiledScaleIndex`. They used to be held out of it, and
`op_kernel/common/turboquant_strip_layout.h` was this repo's own restatement of
the writers' contract standing against them, because each helper states what the
*decode* reads and a reader might take it for what the *writers* produce. Both
files are gone in favour of a test: the writers' arithmetic is stated for a host
reader as `key_nibble` / `value_nibble` / `scale_index` in
`tests/ut/ops/turboquant/common/turboquant_test_common.hpp`, and
`tests/ut/ops/turboquant/test_layout_bridge.cpp` is a device-free gate that
states both contracts in C++ and checks, over four geometries, that the NZ
writer and an independently re-derived decode reader name the same byte and the
same nibble, that both maps are bijections onto their allocations, that both
scale planes are bijections, and that the two contracts' co-byte pairings are
**structurally incompatible**. A gate says it, so a comment no longer has to —
and the mismatch below is now a measured property rather than a warning.

`routines/`, `include/sim_micro_testbed.hpp`, `op_kernel/common/turboquant_codec.h`
(a one-line compatibility shim) and `op_kernel/common/turboquant_sampling.hpp`
(the removed `ASCEND_MICRO_SPARSE_SAMPLING` probe) are gone: nothing reaches them
after this port.

## The three ops

`bindings/torch_bindings.cpp` registers into `torch.ops.turboquant_ops`, which is
`TURBOQUANT_DEFAULT_NAMESPACE` in `vllm_ascend/attention/turboquant_v1.py`.
The Python adapter resolves each op by name, so an unregistered op — any build
that is not ascend950 — makes it warn once and run its PyTorch golden model.
That is the whole fallback contract, and `TURBOQUANT_KERNEL_STAGES` selects per
stage at runtime with no recompilation.

| Op | Launcher | Host derivation |
| --- | --- | --- |
| `npu_turboquant_reshape_and_cache_a8` | `launch_turboquant_reshape_and_cache_bf16` | `tokens_per_core = 16`, `num_tokens % 16 == 0`, grid `num_tokens / 16` (the launcher's own) |
| `npu_turboquant_reshape_scatter_aiv` | `launch_turboquant_reshape_scatter_aiv` | `total_items = num_requests * num_kv_heads`, `item_begin = 0`, `item_count = total_items`, `block_dim = min(total_items, aiv_core_num)`, `cu_seqlens` null |
| `npu_turboquant_fused_decode` | `turboquant_mm_fused_decode_opt_raw_query_impl` | the host tiling descriptors, the Split-K workspace, the fp32 `query_rot` image, the task split, and the stream; bf16 query enforced by `TORCH_CHECK` |

Each schema takes tensors and the few scalars that are not derivable from them.
`num_tokens`, `num_kv_heads`, `head_size`, `block_size` and `num_blocks` come
off the activation and cache shapes; a schema that also took them would let a
caller contradict the tensors it passed.

All three have Meta registrations, so a `torch.export` or FakeTensor trace walks
them instead of trying to run them. Nothing in the served path needs those:
both TurboQuant entry points are graph splitting ops, so under PIECEWISE the
kernels launch eagerly between captured pieces. A full graph is refused
elsewhere (`NPUPlatform` clamps the graph mode). Decode tiling is built from
the CPU sequence-length mirror in the metadata builder, outside the per-layer
decode launch; the launch does not read context lengths back from the device.

The decode and tiling-builder schemas both accept an optional trailing
`fused_context_limit` scalar (default `4096`, valid range `0..INT32_MAX`). An
explicit override must be identical in the builder and launch. The compact
live-write NPU tests use `192` with contexts up to 513 tokens to enter the real
Split-K reduction path; production metadata keeps the default. Rebuild the
extension when updating Python callers to this schema.

### Serving mode: eager execution is a debug tool, not a configuration

Run TurboQuant with **eager mode disabled** (the default): do not pass
`--enforce-eager` / do not set `enforce_eager=True`, and leave
`compilation_config` at its default so the engine runs `VLLM_COMPILE` with
`CUDAGraphMode.PIECEWISE`. The TurboQuant entry ops are graph-splitting, so the
attention pieces around them are captured and replayed while the kernels
themselves launch eagerly between pieces.

This matters because the fused-decode fast path carries per-step host work
that only graph capture can amortise: the metadata builder's context-lens
staging ring, the host C++ `npu_turboquant_build_decode_tiling` walk plus its
host-to-device upload, and the per-layer Python dispatch in
`vllm_ascend/attention/turboquant_v1.py`. Under `--enforce-eager` every decode
step pays all of that on the critical path — plus eager-mode Python overhead
around every operator — and the result looks like a regression even when all
of `TURBOQUANT_KERNEL_STAGES` are enabled and registered. A captured decode
replays the whole launch sequence; only the metadata build and the tiling
upload stay on the host, once per step for the whole model.

Also keep the diagnostic switches off when measuring:
`VLLM_ASCEND_TURBOQUANT_TRACE=0` (dispatch logging walks every tensor's shape,
stride and layout per call) and `TURBOQUANT_PROFILE=0` (a profiler context per
layer per step). Both are off by default.

### Host work and fallback paths

The following is a source audit, not an NPU latency measurement. The fused
decode custom op launches Ascend C directly; its registered path does not run
the reference rotation, gather, or dequantization operators.

| Path | Host work or synchronization |
| --- | --- |
| Decode metadata | Host C++ builds descriptors once per cache-group metadata build. Tiling uploads use a fresh pinned allocation and a nonblocking `copy_` on the current stream. The NPU host allocator retains the source until its copy completes. |
| Context lengths | The CPU mirror is uploaded asynchronously when pinning succeeds. Before reusing a host ring slot, an event query checks completion; a busy slot gets a replacement allocation without waiting for the device. CPU-only/unpinned staging uses a blocking copy. |
| Missing or disabled decode op | Decode falls through to PyTorch gather/dequantization, then NPU fused inference attention. Block-table min/max validation reads device scalars on the host. |
| Prefill cache ingestion | The strip planner calls `slot_mapping.tolist()`: this is a blocking device-to-host readback, cached once per metadata object across layers. Partial-strip waves also construct device index tensors from host indices. |
| Prefill without a cache hit | Attention uses the original K/V tensors and NPU fused inference attention; cache ingestion still has the planner cost above. |
| Cache-hit or mixed prefill | The cache is gathered/dequantized with PyTorch operators on its device before NPU fused inference attention. Query row indices and query lengths share one pinned upload cached across layers. Dense cache reconstruction and its block-table validation still happen per layer. |
| Reference cache writer | Boolean validity checks and conversion of the maximum slot to a Python integer read device scalars on the host. Boolean indexing can add dynamic-shape synchronization. |
| Optional slot validation | `VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS=1` makes the C++ batch writer copy slots to CPU for validation. Leave it disabled for throughput measurements. |
| First decode | The C++ basis pool and mode tables have blocking uploads on their first use, then are cached by device and geometry. Warm up before measuring. |

`TURBOQUANT_KERNEL_STAGES` defaults to `none`. For a kernel measurement,
enable `reshape_and_cache,scatter,decode` and check that all three ops are
registered in the selected namespace. Missing ops log fallback warnings;
`VLLM_ASCEND_TURBOQUANT_TRACE=1` logs the selected branch and tensor devices.
PyTorch reference execution on NPU tensors is still NPU tensor computation,
but the scalar checks above synchronize with the host.

The previous tiling upload used pageable memory and `non_blocking=false`,
which reaches a stream synchronization in
[torch-npu's copy implementation](https://github.com/Ascend/pytorch/blob/v2.10.0/torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp).
The asynchronous upload removes that wait. It does not remove host descriptor
construction or the remaining prefill readback, and no speedup is claimed
without device measurements.

Run the transfer benchmark on an ascend950 build, both idle and with queued
compute, to separate host submission time from elapsed time including drain:

```bash
python benchmarks/scripts/benchmark_turboquant_tiling.py --batch-sizes 1 16 128
python benchmarks/scripts/benchmark_turboquant_tiling.py --queued-matmuls 4
pytest -sv tests/ut/attention/test_turboquant_tiling_npu.py
```

The benchmark compares the former CPU-build/blocking-upload sequence with the
pinned asynchronous binding, after warmup. The NPU tests compare queued
descriptor snapshots against the CPU builder, including non-default stream
ordering, source lifetime under allocator churn, invalid lengths, and context
staging reuse. Use an enclosing torch-npu profiler around several serving
steps to rank the remaining prefill work and kernel execution; per-layer
`TURBOQUANT_PROFILE=1` contexts perturb latency and should only diagnose paths.

## The cache formats now meet

This section used to be "the two cache formats do not meet", and it was the
most important thing to know before switching a stage on: **both** writers
produced the RFC 3.1 strip cache while the fused decode reads the NZ-tiled
cache of `NzTiledPackedByte()` in `common/turboquant_layout.h`. Both
writers are built with `TURBOQUANT_INGRESS_NZ=1` now (the define lives in the
root `CMakeLists.txt`, next to `ASCEND_MICRO_CANN_LAUNCH`):

* the **batch writer**'s NZ mode adds an in-UB NZ stage (`NzTokenMajor`,
  `NzGroupGather`, the weave, and the strided `DataCopyPad` stores) behind the
  same rotation/quantisation pipeline;
* the **scatter writer**'s NZ mode deletes more than it adds: the NZ pairing is
  token-major, one byte wholly owned by one (token, kv head), so the value
  plane's 2048-byte strip read-modify-write disappears and both planes become
  `kNzGroups` bursts of 32 contiguous bytes per token plus one lane of the
  slot-major scale record. The `MTE3_MTE2` strip-recycle edge goes with them.

Both writers' open-coded NZ addressing is tied back to `turboquant_layout.h` by
`static_assert` (`NzStripAgrees` / `NzAgrees`), so a writer that drifted from
the reader cannot compile. `block_size` must be a multiple of 64 for the
writers and a power-of-two multiple of 64 for the decode's closed-form tile
cursor, so 64, 128, 256, ... serve all three.

The strip cache still exists in three places, none of them a serving path: the
`TURBOQUANT_INGRESS_NZ=0` build variant (uncompiled here), the golden model's
`CHANNEL_MAJOR` golden layout, and
`vllm_ascend/attention/turboquant_nz_ingress.py`, the reference strip-to-NZ
repack for caches written before the ingress existed. The record of *why* the
two formats could never share a cache — a strip byte holds ONE channel at TWO
tokens, an NZ byte ONE token at channels `col` and `col + 128`, and at
`block_size` 64 exactly 10 of 32768 addresses agree — is kept in
`tests/ut/ops/turboquant/test_layout_bridge.cpp`, which still gates that the NZ
writer arithmetic and an independently re-derived decode reader name the same
byte and the same nibble.

The Python side knows about the writers' layout: `TurboQuantValueLayout` and
`TurboQuantOpsAdapter.value_layout()` derive it from which writer owns the page
(`NZ_TILED` once either NPU writer is staged), and the golden gather reads an
NZ cache back so that a mixed configuration — NPU writer, PyTorch reader —
still reconciles.

## Gates that need no NPU

```bash
pytest --noconftest tests/ut/ops/test_turboquant_contract_gates.py
```

Two gates, both toolkit-free — the layout contract, the host tiling and the
kernels' host-visible geometry and nothing else, so a build plus a run is a
couple of seconds with any C++17 compiler and needs no CANN and no NPU.

* `tests/ut/ops/turboquant/test_contract_gates.cpp` — the fused decode's UB
  arena `static_assert`s; the host tiling's dispatch odometer and descriptors
  against the arithmetic they replace, over thirteen geometries plus five
  refusals; seven launch-guard admissions and refusals; the mode-2 channel map;
  the shared online-softmax phase order; and a **marked-graph token-balance
  audit of the seamless chunk pipeline** over 1, 2, 3, 4, 8, 17 and 64 strips —
  every ring stays inside its depth, every primed token reaches a terminal
  sink, and a 64-strip share costs exactly 64x the chunk work of a 1-strip
  share with no extra prologue, drain or rendezvous per strip, which is the
  seamless property stated as a count. It `#include`s
  `ingress/native/turboquant_reshape_batch.cpp` directly: the device body is behind
  `#if defined(__CCE_AICORE__)`, so a host compiler sees the arena constants and
  their `static_assert`s, which is the point.
* `tests/ut/ops/turboquant/test_layout_bridge.cpp` — the NZ/strip layout bridge
  described under "What is not mirrored".

The token-balance audit is a **counting argument**, not a deadlock proof and not
a liveness or overlap claim: source order and balanced events do not establish
overlap, which needs a trace with intersecting MTE2/Cube intervals. The
seamless pipeline has **no CAModel evidence** — the isolated head/middle/tail
page canaries, the depth and wrap tests, and the trace audit and census are all
unstarted. The NPU legs in
`tests/e2e/pull_request/one_card/turboquant/` have never been linked against the
kernel libraries.

The three kernel translation units also build for `dav-3510` with no Torch and
no driver, through a throwaway CMake project whose only target is

```cmake
ascendc_library(tq_gate SHARED ingress/native/turboquant_reshape_batch.cpp ingress/native/turboquant_reshape_scatter.cpp decode/native/turboquant_fused_decode.cpp)
```

configured with `-DSOC_VERSION=Ascend950PR_9599`, the `Unix Makefiles`
generator, and the same two defines the root `CMakeLists.txt` passes.

**Which `ascendc_library` you get decides everything else here.** CANN 9.2 ships
two, and the name is rebound by whichever module set was included last:

| include | implementation | what it does |
| --- | --- | --- |
| `ascendc_kernel_cmake/ascendc.cmake` | `legacy_modules/function.cmake` | compiles each TU several times through `ExternalProject_Add`, then generates `auto_gen/<target>/host_stub.cpp` and the `aclrtlaunch_*` stubs from the device objects |
| `ascendc_kernel_cmake/asc-config.cmake` + `enable_language(ASC)` | `asc_modules/FindASC.cmake` | `set_source_files_properties(LANGUAGE ASC)` plus a plain `add_library`; one `bisheng` invocation per TU does the host pass and both device passes, and generates nothing |

The root `CMakeLists.txt` takes the **ASC** path on `ascend950` and fails the
configure if `asc-config.cmake` is absent, so that is the one the wheel builds
with. A standalone gate that includes only `ascendc.cmake` gets the legacy one
and hits all five traps below; the ASC path hits none of them — it is an ordinary
`add_library`, so it rebuilds incrementally, needs no `llvm-objdump` on `PATH`,
runs no merge script and does not glob objects. Both toolchains link these
sources, so either makes a usable gate; just do not read a trap list written for
one as if it applied to the other.

Five things make the legacy gate work:

* `ASCEND_MICRO_CANN_LAUNCH` and `ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1` have to
  be defined, through **both** `ascendc_compile_definitions` and a single
  `ascendc_compile_options -forward-options-to-host-compiler -D...` — see "The
  define you cannot omit" below. Only one `ascendc_compile_options` call per
  target: the device half appends its options while the host half assigns them,
  so a second call silently discards the first's host-forwarded defines.
* `$ASCEND_HOME_PATH/tools/ccec_compiler/bin` has to be on `PATH`. The
  preprocess step shells out to `llvm-objdump` by bare name and dies late
  without it.
* `CMAKE_BUILD_TYPE` has to be set. CANN's `merge_mix_obj.sh` runs under
  `set -e` and the generated recipe interpolates `${CMAKE_BUILD_TYPE}` after a
  bare `--build-type`; with it empty the option has no argument, the script's
  last `shift 2` runs off the end and returns non-zero, and `set -e` kills it
  with no message. The build then dies at `<target>_merge_obj` with nothing but
  `Error 1`, at 95%, after every kernel compiled clean. This repo's root
  `CMakeLists.txt` defaults it, so only a throwaway project hits it.
* The build directory must not contain a dot-component. CANN's object glob never
  descends into one, and the failure names a missing file that exists.
* It must be fresh every time. `ascendc_library` has no incremental rebuild,
  because its preprocess step feeds `ld.lld` its own previous output.

Under the legacy path the target carries both kernel types: `device.o` holds the
two `MIX_AIC_1_2` kernels and `device_aiv.o` the `AIV_ONLY` scatter writer, which
is why one `ascendc_library` is enough for all three sources.

### The define you cannot omit

Each of the three sources is shaped

```c++
#if defined(__CCE_AICORE__)                        // the device bodies
#endif
extern "C" __global__ __aicore__ void kernel(...)  // the entry points
{ #if defined(__CCE_AICORE__) ... #endif }
#if defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(__CCE_AICORE__)
#endif                                             // the host launchers
```

Every `ascendc_library` compiles each translation unit several times, and the
**host** pass is the one with `__CCE_AICORE__` undefined. Without
`ASCEND_MICRO_CANN_LAUNCH` that pass compiles to nothing past the entry points,
and the three native entry points the Torch wrappers call —
`launch_turboquant_reshape_and_cache_bf16`,
`launch_turboquant_reshape_scatter_aiv` and
`vllm_ascend::turboquant_mm_fused_decode_opt_raw_query_impl` — are simply absent.

Nothing about the library looks wrong when that happens. It builds, the device
merge succeeds, every kernel is present, and the generated
`aclrtlaunch_turboquant_*` stubs are there too. The only symptom is an undefined
reference at the *consumer's* link. A substring search does not catch it either,
because `aclrtlaunch_turboquant_reshape_and_cache_bf16` contains
`launch_turboquant_reshape_and_cache_bf16` — so check these symbols with an
exact match, and demangle for the C++ one.

### Why the entry points are outside the device guard

Under the ASC language toolchain a translation unit is single-source: `bisheng`
runs one host pass and two device passes (`__DAV_C310_CUBE__`, `__DAV_C310_VEC__`)
over the same text, and the host-side launch stub of a `__global__` function is
emitted by the **host pass from that function's definition**. A prototype is not
enough, and nothing generates the stub for you the way `host_stub.cpp` does on
the legacy path.

So the entry-point definitions have to be visible in the host pass, with only
their bodies behind `__CCE_AICORE__`. `__global__`, `__aicore__`, `__gm__`,
`bfloat16_t` and `half` are all available there with no CANN header — they come
from `bisheng`'s force-included `__clang_cce_types.h` — while `GM_ADDR` needs
`kernel_operator.h`, which `turboquant_fused_decode.cpp` therefore includes
unconditionally. Nothing is lost by emptying the body in the host pass: the
kernel *type* is a device-link property (`KERNEL_TASK_TYPE_DEFAULT` lowers to
`__builtin_cce_kernel_type_set`, and omitting it fails the device link with "the
type of kernel function ... auto derivate failed"), not something the host stub
carries.

With the definitions inside the device-only region the build still succeeds and
every `launch_*` is present, so the failure is invisible until Python `dlopen()`s
the extension:

```text
undefined symbol: turboquant_reshape_and_cache_bf16
```

Measured on CANN 9.2.0-beta.2: all seven `__global__` symbols came out `U` in
`libvllm_ascend_turboquant.so` with the definitions inside the guard, and `T`
with them outside it. So gate on the **kernels** as well as the launchers, and
remember that plain `nm` is not enough on the legacy path, which strips the
symbol table — use `nm -D`:

```bash
nm -DC libvllm_ascend_turboquant.so | grep -E "^ *U +(turboquant|launch)"
```

must print nothing.

Anything that launches a kernel needs a device, and the differential against the
golden model for ingestion and decode numerics needs one too.
