# TurboQuant kernels and Torch bindings

Everything in this directory except `torch_bindings.cpp` is a **mirror** of the
`turboquant-core` repository — `include/turboquant/` and `src/` there, the
audited production kernels extracted from
`kernels/ops/turboquant/vllm_integration_layer` in the ascend-micro-harness.

The mirror is byte-for-byte in content but **not** in include paths. The
standalone repo separates normative contracts (`include/turboquant/`) from
implementation (`src/{prefill,scatter,decode,common,host}/`); this tree keeps the
CANN operator layout (`include/`, `op_kernel/{common,cube,vector}/`, `op_host/`,
`op_adapter/`). So every `#include` of a sibling contract is rewritten on the way
in, and nothing else is. `tools`-free and reproducible: the rewrite is a fixed
map from one layout to the other, recorded in the port commit.

Do not edit the mirrored files to fix something here. Fix it in `turboquant-core`,
re-run its own gates, and re-mirror — the Ascend C vector and cube logic, the
pipeline depth, the mode-2 channel map, the cross-core flag budget and the UB
alignments are all load-bearing and audited there.

There is **one** standing exception, and it is build integration rather than
kernel logic: in the three kernel translation units the `__global__` entry-point
*definitions* sit outside `#if defined(__CCE_AICORE__)`, with only their bodies
inside it. The harness builds them with CANN's legacy `ascendc_library`, which
generates the host launch stubs itself; this repo builds them with the ASC
language toolchain, which does not. See "Why the entry points are outside the
device guard" below, and do not let a re-mirror undo it — it is the difference
between a loadable `libvllm_ascend_turboquant.so` and one whose every kernel
symbol is undefined.

## What is mirrored

| Path | What it is |
| --- | --- |
| `op_kernel/turboquant_reshape_batch.cpp` | prefill batch writer, MIX_AIC_1_2, **mode-2 only**, one seamless chunk stream over a block's whole share of strips, built with **`TURBOQUANT_INGRESS_NZ=1`** (root `CMakeLists.txt`): writes the NZ-tiled cache the fused decode reads rather than the RFC 3.1 strip cache |
| `op_kernel/turboquant_reshape_scatter.cpp` | decode scatter writer, AIV only, per-token NZ write (built with `TURBOQUANT_INGRESS_NZ=1` like the batch writer; the strip read-modify-write exists only in the uncompiled `=0` variant) |
| `op_kernel/turboquant_fused_decode.cpp` | fused decode, pre-rotated and raw-query entries, **bf16 only** |
| `op_kernel/common/`, `op_kernel/cube/`, `op_kernel/vector/` | codec, layout, wave protocol, the Cube and Vector services and their micro-routines |
| `include/` | the decode's launch ABI, the GEMM-reduce and Split-K contracts |
| `op_adapter/` | the C declaration surface (`launch_*`, `turboquant_mm_fused_decode_opt_raw_query_impl`) |
| `op_host/turboquant_tiling.{h,cpp}` | the decode's host tiling engine — a declaration and a real translation unit, not an `inline` in a device header |

### What the current port changed

Three things are gone from the kernels, and the seam follows them rather than
papering over them.

* **The prefill is mode-2 only.** There is no `template <bool Mode2>`, no
  `..._mode4_single_shot` entry point and no `reuse_mode4_flags` launcher
  argument. Mode 2 lowers to `ffts_cross_core_sync`: each id is a counter in
  0..15 that banks unconsumed announcements, so recycling one across pipeline
  generations is sound by construction and needs no single-shot allocation to
  bisect against. The three channels are `kSync2PanelReady = 8`,
  `kSync2PanelFree = 9`, `kSync2PageDrained = 10`, single-sourced in
  `op_kernel/common/turboquant_wave_protocol.hpp`, each behind its
  `HardEvent::x_S` drain. `bool reuse_mode4_flags` is therefore out of the
  `npu_turboquant_reshape_and_cache_a8` schema.
* **The prefill pipeline is seamless across strip boundaries.** The outer strip
  loop is gone. A block owns `tokens_per_core` tokens = `tokens_per_core / 16`
  strips and runs them as one chunk stream of `strips * kChunks` columns; column
  *j* stages chunk *j* and multiplies chunk *j − kIngressLookahead*, and both
  indices cross strip boundaries, so MTE2 does not drain at the end of a
  16-token strip. The ring periods all divide `kChunks` (static_asserted), so a
  boundary never shifts which slot a chunk owns. The binding still passes
  `tokens_per_core = 16`; raising it is the knob for a wider share and is
  deliberately not raised yet, because nothing has measured which width is
  fastest on the part.
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
  `op_host/turboquant_tiling.cpp` now, and the binding, the kernel launcher and
  the device-free gate all link the same object. `include/turboquant_fused_decode.hpp`
  is gone, superseded by it.

## What is not mirrored

* `torch_bindings.cpp` — this repo's own Torch registration, see below.

`op_kernel/common/turboquant_layout.h` is mirrored **in full** now, including the
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

`torch_bindings.cpp` registers into `torch.ops.turboquant_ops`, which is
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
elsewhere for an unrelated and more serious reason (`NPUPlatform` clamps the
graph mode), and the fused decode's binding copies the context lengths to the
host to build its tiling, which is itself illegal on a capturing stream.

## The cache formats now meet

This section used to be "the two cache formats do not meet", and it was the
most important thing to know before switching a stage on: **both** writers
produced the RFC 3.1 strip cache while the fused decode reads the NZ-tiled
cache of `NzTiledPackedByte()` in `op_kernel/common/turboquant_layout.h`. Both
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
  `op_kernel/turboquant_reshape_batch.cpp` directly: the device body is behind
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
ascendc_library(tq_gate SHARED op_kernel/turboquant_reshape_batch.cpp op_kernel/turboquant_reshape_scatter.cpp op_kernel/turboquant_fused_decode.cpp)
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

- `ASCEND_MICRO_CANN_LAUNCH` and `ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1` have to
  be defined, through **both** `ascendc_compile_definitions` and a single
  `ascendc_compile_options -forward-options-to-host-compiler -D...` — see "The
  define you cannot omit" below. Only one `ascendc_compile_options` call per
  target: the device half appends its options while the host half assigns them,
  so a second call silently discards the first's host-forwarded defines.
- `$ASCEND_HOME_PATH/tools/ccec_compiler/bin` has to be on `PATH`. The
  preprocess step shells out to `llvm-objdump` by bare name and dies late
  without it.
- `CMAKE_BUILD_TYPE` has to be set. CANN's `merge_mix_obj.sh` runs under
  `set -e` and the generated recipe interpolates `${CMAKE_BUILD_TYPE}` after a
  bare `--build-type`; with it empty the option has no argument, the script's
  last `shift 2` runs off the end and returns non-zero, and `set -e` kills it
  with no message. The build then dies at `<target>_merge_obj` with nothing but
  `Error 1`, at 95%, after every kernel compiled clean. This repo's root
  `CMakeLists.txt` defaults it, so only a throwaway project hits it.
- The build directory must not contain a dot-component. CANN's object glob never
  descends into one, and the failure names a missing file that exists.
- It must be fresh every time. `ascendc_library` has no incremental rebuild,
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
and the three entry points `torch_bindings.cpp` calls —
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

```
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
