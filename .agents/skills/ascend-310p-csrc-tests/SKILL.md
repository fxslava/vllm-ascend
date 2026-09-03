---
name: ascend-310p-csrc-tests
description: "Build, run and profile the bare-metal C++ GTest suite in csrc/tests for the Qwen3.5 kernels (RMSNorm, RoPE, SwiGLU, PagedAttention) on Ascend 310P3, inside the official CANN Docker container. Covers the exact build, ctest and msprof commands, the environment lockdown rules, and the known aclnn operator availability."
---

# Ascend 310P3 csrc/tests

## Overview

`csrc/tests/` is a standalone GTest suite for the four operator families a
Qwen3.5 forward pass needs on Ascend 310P3. It builds and runs with no Python,
no PyTorch, no `torch_npu` and no Torch C++ ABI, driving CANN directly.

It is **not** part of the repository root `CMakeLists.txt` build. The root
project requires Torch 2.10.0 and pybind11; this one deliberately does not.
Configure it as its own project.

## Hard constraints

- **Never install, update or download packages** for the *native* x86_64 build
  workflow below. No `apt-get`, `pip`, `conda`, `yum`, no `docker pull`.
  Treat the container and host as read-only by default.
- **Never modify the container or host environment** without explicit,
  per-task approval from the user.
- **Exception, granted 2026-09-03:** the aarch64 cross-compilation setup in
  [Cross-compiling for aarch64](#cross-compiling-for-aarch64) was explicitly
  authorised and has already been performed. It installed the cross toolchain
  into the running container and pulled the arm64 CANN image. That work is
  done; do not redo it, and do not read it as blanket permission to install
  anything else.
- **If a dependency, compiler tool, library or operator symbol is missing:
  STOP.** Do not look for substitutes, do not rename symbols, do not vendor a
  replacement. Report the exact missing component plus the error log and hand
  control back.
- Scope of edits is `csrc/tests/` only. Do not touch root project files or core
  Python modules.
- Target SoC is `ascend310p3` only. No 910B, no native BF16, no MLA.

## Environment

Verified facts about this workstation. Re-check them before assuming a command
will work elsewhere.

| Fact | Value |
| --- | --- |
| Docker on the Windows host | **not installed** |
| Docker location | inside WSL 2 `Ubuntu-22.04`, `/usr/bin/docker`, server 29.7.2 |
| Docker socket permissions | WSL user `dmin` is **not** in the `docker` group and `sudo` needs a password; run as `wsl -u root` |
| Image | `swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-310p-ubuntu22.04-py3.10`, already cached locally (33.2 GB) |
| Repo path in WSL | `/mnt/c/vllm-ascend` |
| Repo path in container | `/vllm-workspace/vllm-ascend` |
| CANN in container | `/usr/local/Ascend/cann-9.1.0` |
| Toolchain in container | cmake 3.22.1, g++ 11.4.0, ninja 1.10.1 |
| Cross toolchain in container | `aarch64-linux-gnu-g++` 11.4.0 (installed 2026-09-03) |
| aarch64 CANN copy | `/vllm-workspace/vllm-ascend/cann_aarch64`, 1.6 GB, gitignored |
| `file` inside the container | **not installed** — run `file` from WSL against `/mnt/c/...` |

There are **two** containers running this image. Select by mount source, not by
name: only the one bound `/mnt/c/vllm-ascend -> /vllm-workspace/vllm-ascend` is
this repo. The other maps `C:\Users\Admin\Documents\vllm-ascend` to the same
container path and will silently build the wrong tree.

```bash
wsl -d Ubuntu-22.04 -u root -- docker ps -q --filter volume=/vllm-workspace/vllm-ascend
```

Every command below is run from Windows as:

```bash
wsl -d Ubuntu-22.04 -u root -- bash -lc '<command>'
```

Running `docker` from Git Bash or PowerShell on the host fails with
`docker: command not found`. Running it as the default WSL user fails with
`permission denied ... unix:///var/run/docker.sock`. Neither is a reason to
install or reconfigure anything: use `-u root`.

## 1. Build

```bash
wsl -d Ubuntu-22.04 -u root -- bash -lc 'docker run --rm -v /mnt/c/vllm-ascend:/vllm-workspace/vllm-ascend -w /vllm-workspace/vllm-ascend swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-310p-ubuntu22.04-py3.10 bash -lc "source /usr/local/Ascend/ascend-toolkit/set_env.sh && git config --global --add safe.directory /vllm-workspace/vllm-ascend && export SOC_VERSION=ascend310p3 && export SETUPTOOLS_SCM_PRETEND_VERSION=0.7.0.dev0 && cmake -S csrc/tests -B build/csrc-tests -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=ascend310p3 && cmake --build build/csrc-tests -j\$(nproc)"'
```

Expected configure output, which doubles as a health check:

```
-- vllm-ascend tests: CANN at /usr/local/Ascend/cann-9.1.0
-- vllm-ascend tests: target SoC ascend310p3
-- vllm-ascend tests: linking /usr/local/Ascend/cann-9.1.0/lib64/libacl_op_compiler.so
-- vllm-ascend tests: linking /usr/local/Ascend/cann-9.1.0/lib64/libnnopbase.so
-- vllm-ascend tests: linking /usr/local/Ascend/cann-9.1.0/lib64/libopapi.so
-- vllm-ascend tests: vllm_ascend_kernels not found (expected on ascend310p)
```

The last line is correct, not a warning to fix. The root `CMakeLists.txt:74`
skips `vllm_ascend_kernels` on `ascend310p`, and the four kernels under test
come from the CANN op-api rather than from `csrc/`.

Build produces four binaries in `build/csrc-tests/`:
`test_rmsnorm_310p`, `test_rotary_embedding_310p`,
`test_activation_swiglu_310p`, `test_paged_attention_310p`.

### googletest and the no-download rule

A **fresh** configure runs `FetchContent` against `github.com`, which the
lockdown forbids. `build/csrc-tests/_deps/googletest-src` is already populated,
so reusing the existing build directory performs no download. If the build
directory has to be recreated, pass one of these instead of allowing the fetch:

```bash
-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest
```

```bash
-DVLLM_ASCEND_TESTS_FETCH_GTEST=OFF
```

`-DVLLM_ASCEND_TESTS_FETCH_GTEST=OFF` requires a system googletest
(`find_package(GTest)`). If neither a local source tree nor a system install is
present, that is a missing dependency: **stop and report it**.

## 2. Host-only tests

These need no NPU. They cover the CPU reference implementations, the fp16
conversion, the 310P NZ cache layout arithmetic and the shape/alignment
contracts.

```bash
wsl -d Ubuntu-22.04 -u root -- bash -lc 'docker run --rm -v /mnt/c/vllm-ascend:/vllm-workspace/vllm-ascend -w /vllm-workspace/vllm-ascend swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-310p-ubuntu22.04-py3.10 bash -lc "source /usr/local/Ascend/ascend-toolkit/set_env.sh && cd build/csrc-tests && for b in test_rmsnorm_310p test_rotary_embedding_310p test_activation_swiglu_310p test_paged_attention_310p; do ./\$b --gtest_filter=-Qwen35/* || exit 1; done"'
```

Expect **17 passed, 0 failed, 0 skipped** (3 + 4 + 4 + 6). Verified on CANN 9.1.0.

### Do not use `ctest -R 'Reference|Shapes'`

That filter is wrong in both directions:

- It **misses** `PagedKvLayout.MatchesGetKvCacheShapeOn310P` and
  `PagedKvLayout.OffsetsAreUniqueAndInBounds`, which match neither pattern.
- It **picks up** every device test named `MatchesCpuReference`, which then
  report as `Skipped` and pad the run to 63 entries.

The device suites are exactly the parameterised ones under `Qwen35/`, so
`--gtest_filter=-Qwen35/*` selects the host-only set precisely. If a `ctest`
run is required for reporting, this is the equivalent:

```bash
ctest --test-dir build/csrc-tests -E 'Qwen35' --output-on-failure
```

## 3. On-device tests

Requires a real Ascend 310P3 card. Adjust `DEVICE` to the card in use. Device
and driver flags follow `docs/source/developer_guide/contribution/testing.md`.

```bash
wsl -d Ubuntu-22.04 -u root -- bash -lc 'docker run --rm --shm-size=1g --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info -v /etc/ascend_install.info:/etc/ascend_install.info -v /mnt/c/vllm-ascend:/vllm-workspace/vllm-ascend -w /vllm-workspace/vllm-ascend swr.cn-south-1.myhuaweicloud.com/ascendhub/cann:9.1.0-310p-ubuntu22.04-py3.10 bash -lc "source /usr/local/Ascend/ascend-toolkit/set_env.sh && ctest --test-dir build/csrc-tests --output-on-failure"'
```

Select a non-zero device ordinal with `ASCEND_TEST_DEVICE_ID` (the harness reads
it; the default is 0):

```bash
-e ASCEND_TEST_DEVICE_ID=3
```

Each binary prints an operator inventory before the first test. Read it first:
a run that ends in skips explains itself on line 2.

### Interpreting a skip

| Skip message | Meaning |
| --- | --- |
| `No usable Ascend device: ... aclInit ... status 500000` | container has no card mapped; add the `--device` flags |
| `Test targets Ascend 310P; attached device reports '<soc>'` | wrong part; the suite is 310P-only by design |
| `aclnn operator <name> is not exported by this CANN install` | see the operator status table below — **stop, do not substitute** |

## 4. Profiling with msprof

Always narrow to a single parameterised case. Without a filter the trace covers
every shape in the suite and per-kernel timings cannot be attributed.

Full trace with AI Core pipe utilisation:

```bash
msprof --application="./build/csrc-tests/test_rmsnorm_310p --gtest_filter=*tokens128_hidden8192*" --output=./prof/rmsnorm --aic-metrics=PipeUtilization --ai-core=on
```

Cheaper task-level timeline:

```bash
msprof --application="./build/csrc-tests/test_paged_attention_310p --gtest_filter=*seq4_h32_kv8_d128_b64_ctx512*" --output=./prof/pa --task-time=on --aicpu=on
```

Memory-access view:

```bash
msprof --application="./build/csrc-tests/test_activation_swiglu_310p --gtest_filter=*tokens128_intermediate14336*" --output=./prof/swiglu --aic-metrics=MemoryUB
```

Export to CSV:

```bash
msprof --export=on --output=./prof/rmsnorm
```

Read `prof/<name>/**/mindstudio_profiler_output/op_summary_*.csv` for per-operator
duration and `op_statistic_*.csv` for the aggregate.

Two caveats when reading the numbers:

- `RunAclnn` synchronises the stream on every call, so timings include launch
  overhead. Use `--gtest_repeat=N` for a stable per-call figure.
- The workspace is allocated and freed per call. In the plugin it comes from the
  caching allocator, so a profile taken here overstates allocation cost.

## aclnn operator status on CANN 9.1.0

Measured by the built-in inventory on
`cann:9.1.0-310p-ubuntu22.04-py3.10`. The suite resolves operators via
`dlopen`/`dlsym` on `libopapi.so`, the same mechanism as
`csrc/aclnn_torch_adapter/op_api_common.h`.

| Operator | Status | Used by |
| --- | --- | --- |
| `aclnnRmsNorm` | found | `test_rmsnorm_310p` |
| `aclnnSwiGlu` | found | `test_activation_swiglu_310p` |
| `aclnnApplyRotaryPosEmbV2` | found | `test_rotary_embedding_310p` |
| `aclnnApplyRotaryPosEmb` | found | fallback for the above |
| `aclnnScatterPaKvCache` | found | `test_paged_attention_310p` (KV write) |
| `aclnnIncreFlashAttentionV4` | found | declared and verified, not yet wired |
| `aclnnRotaryMul` | **MISSING** | GE graph op, not aclnn |
| `aclnnReshapeAndCache` | **MISSING** | ATB op, not aclnn |
| `aclnnPagedAttention` | **MISSING** | ATB op, not aclnn |

### Why three are missing

A full-depth `nm -D` sweep of `/usr/local/Ascend` (no `maxdepth` — ATB sits at
depth 7 and a shallower search misses it) located them outside aclnn entirely:

- `PagedAttention` and `ReshapeAndCache` are **ATB** operators in
  `/usr/local/Ascend/nnal/atb/9.1.0/atb/cxx_abi_{0,1}/lib/libatb.so`
  (`atb::PagedAttentionOperation`, `atb::ReshapeAndCacheOperation`, created via
  `atb::CreateOperation<atb::infer::…Param>`), with tiling in
  `libatb_mixops.so`. ATB is a C++ object API (Operation + VariantPack +
  Context), not the aclnn two-phase C API, so `RunAclnn` cannot drive it.
  `torch_npu._npu_paged_attention` and `_npu_reshape_and_cache` go through ATB.
- `RotaryMul` exists only as the GE graph op `ge::op::RotaryMul` in the legacy
  ONNX/TensorFlow plugin libraries.

The aclnn routes to the same results are `aclnnScatterPaKvCache` (the KV write,
which `BaseDeviceAdaptor.reshape_and_cache` uses via
`npu_scatter_pa_kv_cache`) and `aclnnIncreFlashAttentionV4` (paged decode, with
`blocktable` + `blockSize`). Both are declared with header-verified prototypes
in `csrc/tests/common/aclnn_ops.hpp`.

**Still open, do not guess:** IFA V4 takes key/value as `aclTensorList` and
expects a paged KV layout that is not the 310P 5-D NZ shape the plugin
allocates, so the decode test is not wired to it. Likewise the `cacheMode`
string `aclnnScatterPaKvCache` needs for an NZ cache is unconfirmed
(`"Norm"` is what the ND path uses). Both need a hardware run to settle.

**Any new `MISSING` entry is a stop condition.** Do not guess alternative symbol
names or install anything. Report it and hand back control.

## Operator prototypes are hand-declared

`csrc/tests/common/aclnn_ops.hpp` is the only version-sensitive file. Because
operators are resolved with `dlsym`, the compiler cannot check their argument
lists. Each declaration names the CANN header to verify against and the
`torch_npu` entry point plus vllm-ascend call site it mirrors.

A wrong prototype surfaces as a non-zero status from the planning call with the
`aclGetRecentErrMsg` text attached, not as silent corruption. Verify with:

```bash
grep -rA24 'GetWorkspaceSize' $ASCEND_HOME_PATH/include/aclnnop/aclnn_rms_norm.h $ASCEND_HOME_PATH/include/aclnnop/aclnn_swi_glu.h
```

## 310P specifics the suite encodes

Taken from the plugin source, and several differ from GPU defaults:

- KV cache is 5-D NZ: `(2, num_blocks, num_kv_heads * head_size / 16, block_size, 16)`,
  allocated `ACL_FORMAT_FRACTAL_NZ`. Requires `num_kv_heads * head_size % 16 == 0`.
- Block size is **64 or 128**, not 16/32
  (`AscendAttentionBackend310.get_supported_kernel_block_sizes`), and
  `block_size * head_size <= 128 * 128`.
- SwiGLU needs `x.shape[-1] % 32 == 0`, else `AscendSiluAndMul310` falls back to
  eager torch and the kernel is never reached.
- Rotary embedding supports head dims 64 and 128 only.
- Device allocations are padded to 32 bytes for MTE2/MTE3 burst safety.

## Cross-compiling for aarch64

The 310P3 host CPU is aarch64; the development container is x86_64. Cross
builds use `cmake/aarch64-toolchain.cmake` plus a second, aarch64 copy of the
CANN toolkit at `cann_aarch64/` (gitignored).

### Where the aarch64 toolkit came from

The tag `cann:9.1.0-310p-ubuntu22.04-arm64` **does not exist** in the registry.
`9.1.0-310p-ubuntu22.04-py3.10` is a multi-arch manifest list carrying both
architectures, so the arm64 build is pulled **by digest**. Pulling
`--platform linux/arm64` by *tag* would repoint the local tag at arm64 and
disturb the running x86_64 containers; pulling by digest does not.

```bash
wsl -d Ubuntu-22.04 -u root -- docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/cann@sha256:4bb08628cbefdfb465ff8aa826517a012665a3d73a4c633812e9c81b3fdf460e
```

`build/cross-smoke/extract_cann.sh` does the extraction (`docker create` +
`docker cp` of `latest/aarch64-linux/.`; no qemu needed, since nothing arm64 is
executed). It takes ~6 min for 1.6 GB over drvfs. Re-run only if
`cann_aarch64/` is lost.

### Building

```bash
wsl -d Ubuntu-22.04 -u root -- docker exec <cid> bash -c 'cd /vllm-workspace/vllm-ascend && cmake -S csrc/tests -B build/csrc-tests-aarch64 -G Ninja -DCMAKE_TOOLCHAIN_FILE=/vllm-workspace/vllm-ascend/cmake/aarch64-toolchain.cmake -DSOC_VERSION=ascend310p3 && cmake --build build/csrc-tests-aarch64'
```

The toolchain forces `ASCEND_HOME_PATH` to `cann_aarch64`, so the CANN lookup
at the top of `csrc/tests/CMakeLists.txt` resolves to the aarch64 tree with no
edits. A cross build of the suite additionally needs googletest cross-compiled:
pass `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<path>` to reuse the existing source
tree rather than fetching.

### Three things that break a naive toolchain file

- **`try_compile` runs its probe.** Cross builds cannot execute the result, so
  the toolchain sets `CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY`.
- **Driver symbols are not in the toolkit.** `libascendcl.so` →
  `libruntime.so`/`libascend_dump.so` → `drvHdc*`, `hal*`, which live in the
  *driver*, not the toolkit. CANN ships link-time stubs at
  `cann_aarch64/devlib/linux/aarch64/`; without that on `-Wl,-rpath-link` the
  link dies on `undefined reference to drvHdcGetCapacity`. The stubs are
  link-time only and must never shadow the real driver at runtime.
- **Host libraries can satisfy `find_library`.** `CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY`
  keeps the x86_64 toolkit under `/usr/local/Ascend` from being picked up.

### Smoke test

`build/cross-smoke/` is a minimal binary that links `libascendcl.so` and
references real symbols, so the link genuinely resolves.

```bash
wsl -d Ubuntu-22.04 -u root -- docker exec <cid> bash -c 'cd /vllm-workspace/vllm-ascend && cmake -S build/cross-smoke -B build/cross-smoke-build -G Ninja -DCMAKE_TOOLCHAIN_FILE=/vllm-workspace/vllm-ascend/cmake/aarch64-toolchain.cmake -DSOC_VERSION=ascend310p3 && cmake --build build/cross-smoke-build'
wsl -d Ubuntu-22.04 -u root -- file /mnt/c/vllm-ascend/build/cross-smoke-build/cann_aarch64_smoke
```

Expected: `ELF 64-bit LSB pie executable, ARM aarch64`. Run `file` from WSL —
it is not installed in the container.

### Known wart: RUNPATH

Linking a library by absolute path makes CMake record a build-tree RPATH, so
cross-built binaries carry `RUNPATH=/vllm-workspace/vllm-ascend/cann_aarch64/lib64`
— a build-host path that does not exist on the device. Harmless (the loader
falls through to `LD_LIBRARY_PATH` from `set_env.sh`), but strip it for
deployable artifacts with `-DCMAKE_SKIP_BUILD_RPATH=ON` (verified: removes
RUNPATH, keeps `NEEDED libascendcl.so`).

### Not yet done

Nothing has been **run** on aarch64. These binaries have only been built and
inspected; no 310P3 device execution has been attempted from here.

## Verification checklist

- [ ] Configure prints the three CANN libs and the expected `vllm_ascend_kernels not found` line.
- [ ] Build completes with zero warnings under `-Wall -Wextra`.
- [ ] Four binaries exist in `build/csrc-tests/`.
- [ ] Host-only run reports 17 passed, 0 failed, 0 skipped.
- [ ] Operator inventory matches the status table above; any new `MISSING` entry is a stop condition.
- [ ] For cross builds: `file` reports `ELF 64-bit LSB ... ARM aarch64` and `readelf -dW` shows `NEEDED libascendcl.so`.
