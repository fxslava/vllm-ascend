# TurboQuant kernels and Torch bindings

Everything in this directory except `torch_bindings.cpp`,
`op_kernel/common/turboquant_strip_layout.h` and the tail of
`op_kernel/common/turboquant_layout.h` is a **verbatim mirror** of
`kernels/ops/turboquant/vllm_integration_layer` in the ascend-micro-harness.
The mirror is byte-for-byte and needed **no include-path edits**: every include
in the audited tree is either a CANN system header (`kernel_operator.h`,
`tensor_api/tensor.h`) or relative within the subtree, so reproducing the
directory names reproduces the include graph.

Do not edit the mirrored files to fix something here. Fix it in the harness,
re-run its own gates, and re-mirror — the Ascend C vector and cube logic, the
pipeline depth, the mode-4 event pairings, the cross-core flag budget and the
UB alignments are all load-bearing and audited there.

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
| `op_kernel/turboquant_reshape_batch.cpp` | prefill batch writer, MIX_AIC_1_2, four-stage ring over one 16-token strip |
| `op_kernel/turboquant_reshape_scatter.cpp` | decode scatter writer, AIV only, per-token strip read-modify-write |
| `op_kernel/turboquant_fused_decode.cpp` | fused decode, pre-rotated and raw-query entries, fp16 and bf16 |
| `op_kernel/common/`, `op_kernel/cube/`, `op_kernel/vector/` | codec, layout, wave protocol, the Cube and Vector services and their micro-routines |
| `include/` | the fused decode's host tiling descriptors, its launch ABI, the GEMM-reduce and Split-K contracts |
| `op_adapter/` | the C declaration surface (`launch_*`, `turboquant_mm_fused_decode_opt_raw_query_impl`) and the symbol inventory |
| `op_host/turboquant_tiling.*` | the staged tiling descriptor alias and its default-constructibility gate |
| `routines/` | the Cube H16 and ZN page-staging routines |

## What is not mirrored

* `torch_bindings.cpp` — this repo's own Torch registration, see below.
* `op_kernel/common/turboquant_strip_layout.h` — this repo's host-side
  restatement of the strip cache's addressing, kept because
  `tests/ut/ops/test_turboquant_kernel_contracts.cpp` is a device-free gate that
  checks it against the layout headers. Two documented edits from the version it
  replaced: its include moved with the wave protocol, and `kCacheStripRows` is
  now a literal 16 because the audited wave protocol no longer publishes
  `kReshapeTokensPerStrip`.
* `op_kernel/common/turboquant_layout.h` — mirrored except for the four
  ingress helpers the harness added in its `00cf08c`: `NzTiledChannelByte`,
  `NzTiledChannelNibble`, `NzScaleSlotFloats` and `NzTiledScaleIndex`. They are
  **deliberately not here**, and re-mirroring them is a mistake rather than a
  tidy-up, because each states a contract this repo's writers do not implement:

    * they pair nibbles **woven**, channel `c` with channel `c + headSize/2` in
      one byte. Both writers here pair **adjacent elements**, low nibble even —
      the "Layout contracts" block of `op_kernel/turboquant_reshape_batch.cpp`
      and the nibble-index comment in
      `op_kernel/turboquant_reshape_scatter.cpp`. The harness file says so
      itself: the woven pairing is "NOT the RFC 3.1 strip contract's two
      adjacent channels per byte".
    * `NzTiledScaleIndex` indexes scales per slot, `slot * NzScaleSlotFloats(H)
      + kv * H + kvHead`. The scale plane here is plane-major over pages,
      `((kv * numBlocks + block) * numKvHeads + head) * blockSize + key`.

  `NzTiledPackedByte`, which **is** mirrored and is used by
  `op_kernel/vector/turboquant_vector_service.h` and `op_kernel/common/turboquant_mode.h`,
  is a different thing: it addresses the tile a byte lives in, which kv4fp8 does
  store NZ-tiled (`kStoresNzTiles`). Tile addressing and nibble pairing are
  orthogonal; only the latter is in dispute.

  The adjacent-element pairing is gated by
  `tests/ut/attention/test_turboquant_strip_cache.py::test_strip_write_matches_the_kernel_nibble_formula`
  (its `_nibble` helper, at `block_size` 16, 32 and 64), so adopting the woven
  pairing fails there rather than silently writing a cache the decode misreads.
  The harness's own NZ-ingress work — `TURBOQUANT_INGRESS_NZ`, the mode-2
  rendezvous prefill and the `(c, c + 128)` weave — lives in its **standalone**
  `kernels/ops/turboquant/turboquant_reshape_and_cache_bf16.cpp`, a different and
  larger kernel over a different include graph (`baseline/common/`, the harness
  `routines/` and `tests/common/`), with `block_size` meaning the 16-token strip
  height rather than vLLM's page size. None of it has reached
  `vllm_integration_layer`, so there is nothing to mirror yet and a hand-port is
  a reviewed porting job, not a file copy.
* The harness's `tests/`, `standalone/` and `sim_micro_testbed.hpp` consumers.
  Their gates run in the harness checkout, not here.

## The three ops

`torch_bindings.cpp` registers into `torch.ops.turboquant_ops`, which is
`TURBOQUANT_DEFAULT_NAMESPACE` in `vllm_ascend/attention/turboquant_dry_run.py`.
The Python adapter resolves each op by name, so an unregistered op — any build
that is not ascend950 — makes it warn once and run its PyTorch golden model.
That is the whole fallback contract, and `TURBOQUANT_KERNEL_STAGES` selects per
stage at runtime with no recompilation.

| Op | Launcher | Host derivation |
| --- | --- | --- |
| `npu_turboquant_reshape_and_cache_a8` | `launch_turboquant_reshape_and_cache_bf16` | `tokens_per_core = 16`, `num_tokens % 16 == 0`, grid `num_tokens / 16` (the launcher's own) |
| `npu_turboquant_reshape_scatter_aiv` | `launch_turboquant_reshape_scatter_aiv` | `total_items = num_requests * num_kv_heads`, `item_begin = 0`, `item_count = total_items`, `block_dim = min(total_items, aiv_core_num)`, `cu_seqlens`/`launch_probe` null |
| `npu_turboquant_fused_decode` | `turboquant_mm_fused_decode_opt_raw_query_impl` | the host tiling descriptors, the Split-K workspace, the fp32 `query_rot` image, the task split, and the stream |

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

## The two cache formats do not meet

This is the most important thing to know before switching a stage on, and it is
the harness's Phase 1 position rather than anything introduced here.

* The **two writers** produce the strip cache: page axis
  `(num_blocks, num_kv_heads, block_size, head_size / 2)`, key nibble
  `token * head_size + channel`, value nibble
  `(token / 16) * (16 * head_size) + channel * 16 + (token % 16)` —
  channel-major *inside a 16-token strip*. Both take any `block_size` that is a
  multiple of 16.
* The **fused decode** reads the NZ-tiled cache of
  `NzTiledPackedByte()` in `op_kernel/common/turboquant_layout.h`:
  `[64-row tile][kv head][32-byte column group][tile row][byte]`, and its
  closed-form tile cursor needs `block_size` to be a power-of-two multiple of
  64.

So the two can now share a `block_size` — 64 or 128 — but not a layout, and the
obstacle is numerical rather than dimensional. A byte of the strip plane holds
ONE channel at TWO tokens; a byte of the NZ plane holds ONE token at channels
`col` and `col + 128`. That is a different pairing of nibbles into bytes, not a
permutation of bytes, so no reordering reconciles them: at `block_size` 64 both
occupy the same 8192 bytes per (tile, kv head) and zero of those bytes hold the
same pair. Each op therefore `TORCH_CHECK`s its own addressing and the mismatch
cannot be expressed as a shape check at all — switching on both ingestion and
decode kernels over one cache would pass every check and attend over the wrong
tokens. Merging the formats is kernel work in the harness.

The Python side knows about the writers' layout: `TurboQuantValueLayout` and
`TurboQuantOpsAdapter.value_layout()` derive it from which writer owns the page,
and the golden gather de-interleaves a strip cache so that a mixed
configuration — NPU writer, PyTorch reader — still reads back correctly.

## Gates that need no NPU

```bash
pytest --noconftest tests/ut/ops/test_turboquant_kernel_contracts.py
```

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
