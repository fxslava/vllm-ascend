# Bare-metal kernel tests and benchmarks (Ascend 310P3, CUDA)

A standalone suite for the five operator families a **Qwen3.5** forward pass
needs, driven straight through the CANN runtime. No Python, no PyTorch, no
`torch_npu`, no Torch C++ ABI.

There are two device backends. The default is Ascend 310P3 through CANN. The
other, selected with `-DENABLE_CUDA=ON`, runs hand-written CUDA C kernels on an
NVIDIA GPU so the suite can be developed and run on a host PC with no Ascend
hardware attached - see [CUDA backend](#cuda-backend). Both share the CPU
references, the fp16 conversion, the deterministic data generator, the tolerance
machinery and the Qwen3.5 shape tables; only the device runtime and the kernels
differ.

| Kernel | Test binary | Benchmark binary | Stands in for |
| --- | --- | --- | --- |
| MatMul (cube unit) | `test_matmul_310p` | `bench_matmul_310p` | `torch.nn.functional.linear` — every QKV / o_proj / gate_up / down projection |
| RMSNorm | `test_rmsnorm_310p` | `bench_rmsnorm_310p` | `torch_npu.npu_rms_norm` — `AscendRMSNorm310.forward_oot` |
| Rotary embedding | `test_rotary_embedding_310p` | `bench_rotary_embedding_310p` | `torch_npu.npu_apply_rotary_pos_emb` — `_rope_forward_oot` |
| SiluAndMul (SwiGLU) | `test_activation_swiglu_310p` | `bench_activation_swiglu_310p` | `torch_npu.npu_swiglu` — `AscendSiluAndMul310.forward` |
| Paged attention + KV cache | `test_paged_attention_310p` | `bench_paged_attention_310p` | `torch_npu._npu_paged_attention`, `torch_npu._npu_reshape_and_cache` |

The Python unit tests for these paths (`tests/ut/ops/test_layernorm.py`,
`test_activation.py`, `test_rotary_embedding.py`) mock `torch_npu` out entirely
and assert only on dispatch. This suite is where the numerics are actually
checked, against fp32 CPU references, at the shapes Qwen3.5 uses.

[COVERAGE.md](COVERAGE.md) is the side-by-side audit of the two suites: what
each one covers, where they disagree on how a tensor is verified, and what is
missing from both.

---

## Layout

```
csrc/tests/
├── CMakeLists.txt                       standalone project, not included from the repo root
├── COVERAGE.md                          C++ vs Python coverage and parity audit
├── common/
│   ├── acl_check.hpp                    ACL_CHECK / ASSERT_ACL_OK, with aclGetRecentErrMsg attached
│   ├── aclnn_ops.hpp / .cpp             the version-sensitive aclnn prototypes — read this first
│   ├── aclnn_runtime.hpp / .cpp         dlopen/dlsym loader, aclTensor RAII, two-phase launch
│   ├── bench_main.cpp                   entry point for the bench_* binaries
│   ├── benchmark.hpp / .cpp             plan-once launch, event timing, statistics, reporting
│   ├── cpu_reference.hpp / .cpp         naive fp32 references, shared by BOTH backends
│   ├── device_buffer.hpp                RAII device allocation, 32-byte default, 512 for benchmarks
│   ├── device_tensor.hpp                device buffer + aclTensor descriptor, with host conversions
│   ├── fp16.hpp                         IEEE-754 binary16 conversion, round-to-nearest-even
│   ├── main.cpp                         entry point for the test_* binaries, prints the inventory
│   ├── qwen_shapes.hpp                  Qwen3.5 shapes and the 310P alignment rules
│   ├── random_data.hpp                  deterministic, platform-independent test data
│   ├── tensor_compare.hpp               allclose with a diagnostic report
│   ├── test_harness.hpp / .cpp          AscendTestEnvironment: aclInit, device, context, stream
│   │
│   ├── cuda_check.hpp                   CUDA_CHECK / ASSERT_CUDA_OK, the acl_check.hpp counterpart
│   ├── cuda_runtime.hpp / .cu           CUDADeviceBuffer, CUDADevice, CUDATestEnvironment
│   ├── cuda_device_tensor.hpp           typed device buffers with the fp16 round trip
│   └── cuda_main.cpp                    entry point for the CUDA test_* binaries
└── kernels/
    ├── test_matmul_310p.cpp             bench_matmul_310p.cpp
    ├── test_rmsnorm_310p.cpp            bench_rmsnorm_310p.cpp
    ├── test_rotary_embedding_310p.cpp   bench_rotary_embedding_310p.cpp
    ├── test_activation_swiglu_310p.cpp  bench_activation_swiglu_310p.cpp
    ├── test_paged_attention_310p.cpp    bench_paged_attention_310p.cpp
    └── cuda/
        ├── cuda_kernels.hpp             host-callable launchers, the nvcc/host-compiler seam
        ├── cuda_block_reduce.cuh        __shfl_xor_sync warp and block reductions
        ├── rmsnorm_kernel.cu            test_rmsnorm.cpp
        ├── rotary_kernel.cu             test_rotary_embedding.cpp
        ├── swiglu_kernel.cu             test_activation_swiglu.cpp
        ├── paged_attention_kernel.cu    test_paged_attention.cpp
        └── test_matmul.cpp              cuBLAS; the one operator with no kernel of its own
```

Each test file has two layers. Tests named `*Reference` and `*Shapes` are
host-only: they check the CPU reference and the layout arithmetic and run
anywhere, including on a build machine with no NPU. The rest need a device and
skip with an explanatory message when one is not attached.

---

## Prerequisites

Everything from here to [CUDA backend](#cuda-backend) is the Ascend 310P3 path.

- CANN toolkit (headers at `$ASCEND_HOME_PATH/include/acl/acl.h`)
- CMake >= 3.16 (3.18+ enables per-test `ctest` granularity)
- A C++17 compiler
- An Ascend 310P3 device for the parity tests; the host-only tests need none

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

## Configure and build

```bash
cmake -S csrc/tests -B build/csrc-tests -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build/csrc-tests -j "$(nproc)"
```

On a host without internet access, either point FetchContent at an unpacked
googletest tree:

```bash
cmake -S csrc/tests -B build/csrc-tests -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/opt/googletest
```

or use a system install:

```bash
cmake -S csrc/tests -B build/csrc-tests -DVLLM_ASCEND_TESTS_FETCH_GTEST=OFF
```

If CANN is not on the default path, pass it explicitly:

```bash
cmake -S csrc/tests -B build/csrc-tests -DASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

## Run

```bash
ctest --test-dir build/csrc-tests --output-on-failure
```

One suite at a time:

```bash
ctest --test-dir build/csrc-tests -R rmsnorm --output-on-failure
```

Or run a binary directly, which is the better option while debugging because it
prints the operator inventory first:

```bash
./build/csrc-tests/test_paged_attention_310p --gtest_filter='*seq4_h32_kv8*'
```

Select a device other than 0:

```bash
ASCEND_TEST_DEVICE_ID=3 ./build/csrc-tests/test_rmsnorm_310p
```

Useful GTest flags: `--gtest_list_tests`, `--gtest_repeat=10`,
`--gtest_shuffle`, `--gtest_output=xml:results.xml`.

---

## Benchmarks

The `bench_*` binaries are a separate suite with the same plumbing and a
different question: not "is the answer right" but "how long does it take". They
link `common/bench_main.cpp` instead of `common/main.cpp`, so there is no GTest
in them at all.

```bash
cmake --build build/csrc-tests --target benchmarks -j "$(nproc)"
```

```bash
./build/csrc-tests/bench_matmul_310p
```

They are registered with ctest under the `benchmark` label, so the default run
stays a correctness run:

```bash
ctest --test-dir build/csrc-tests -LE benchmark --output-on-failure
```

```bash
ctest --test-dir build/csrc-tests -L benchmark --output-on-failure
```

A benchmark exits 77 when no usable 310P is attached, which ctest reports as a
skip rather than a failure. `-DVLLM_ASCEND_TESTS_BUILD_BENCHMARKS=OFF` leaves
them out of the build entirely.

### What is inside the timed region

- **Nothing is allocated.** Device buffers, `aclTensor` descriptors and the
  operator workspace are all created during setup. Buffers are requested at
  512-byte alignment (`kBenchmarkAlignBytes`) rather than the 32-byte test
  default, so a measurement is never charged for a buffer starting mid-line.
- **The operator is planned once.** `aclnn` is a two-phase API and the launch
  entry point normally consumes the executor `GetWorkspaceSize` produced.
  `aclSetAclOpExecutorRepeatable` opts out of that, so the timed loop calls only
  the launch function. When a CANN build does not export it, the fallback
  re-plans inside the loop, which is what `torch_npu` does per call, and every
  report says which path ran. `ASCEND_BENCH_REPEATABLE=0` forces the fallback so
  the two can be compared.
- **Nothing synchronises**, except in `host` mode, which exists precisely to
  measure host-visible single-call latency.
- **Warmup runs first** (20 iterations by default) so kernel compilation,
  workspace first-touch and AI Core clock ramp do not land in sample 0.

### Three timing modes, all reported per shape

| Mode | How | Read it for |
| --- | --- | --- |
| `pipelined` | one event pair around a batch of 10 launches, divided by 10 | throughput; TFLOP/s and GB/s are derived from this row |
| `device` | one event pair around each launch, all recorded back to back, one stream sync afterwards | pure device time per launch |
| `host` | `steady_clock` around launch + `aclrtSynchronizeStream` | the latency a decode step actually pays, launch overhead included |

Each mode reports min, median, mean, P95, P99 and standard deviation in
microseconds over 100 iterations. Percentiles are nearest-rank, so at the
default iteration count P99 is the second-largest sample: read it as a tail
indicator, not a number to tune against.

Every case also carries a checksum, taken once after warmup and once after the
last timed iteration. A benchmark whose operator stopped writing its output,
which is the classic failure of a repeatable-executor path, fails instead of
reporting an impressively small number.

### Knobs

| Variable | Default | Effect |
| --- | --- | --- |
| `ASCEND_BENCH_WARMUP` | 20 | warmup iterations |
| `ASCEND_BENCH_ITERS` | 100 | timed iterations per mode |
| `ASCEND_BENCH_BATCH` | 10 | launches per event pair in `pipelined` |
| `ASCEND_BENCH_MODES` | all three | comma-separated subset of `pipelined,device,host` |
| `ASCEND_BENCH_CSV` | unset | write one row per (case, mode) to this path |
| `ASCEND_BENCH_REPEATABLE` | on | `0` forces the re-plan-per-launch path |
| `ASCEND_TEST_DEVICE_ID` | 0 | device ordinal, shared with the tests |

```bash
ASCEND_BENCH_ITERS=500 ASCEND_BENCH_CSV=matmul.csv ./build/csrc-tests/bench_matmul_310p
```

### Shape coverage

The benchmarks deliberately go past the parity matrix, because the shapes that
matter for performance are not the ones that matter for correctness:

- MatMul sweeps M in {1, 32, 128, 512}, so both the decode GEMV and the prefill
  regime are measured, and adds the fused-QKV (N = 6144, 4608) and `down_proj`
  (K = 11008) widths that the parity suite's cartesian product cannot express.
- RMSNorm, SwiGLU and RoPE add a 512-token batch, since at 128 tokens launch
  overhead is still a visible share of a bandwidth figure.
- The paged suite benchmarks `aclnnScatterPaKvCache` with a shuffled slot
  mapping. Decode attention is registered as an explicit skip, for the reason in
  `common/aclnn_ops.hpp`.

See [COVERAGE.md](COVERAGE.md) for what this does and does not close.

---

## CUDA backend

`-DENABLE_CUDA=ON` swaps the device layer for native CUDA C. It exists so the
kernels a Qwen3.5 decode step needs can be developed, run and debugged on a host
PC, without an Ascend 310P3 in the loop.

What it is not: a port of the plugin, and not a second implementation anyone
ships. It is a second *device* for the same tests, so a change to the CPU
references or the test logic is exercised somewhere before it reaches hardware.

| Kernel | Test binary | Notes |
| --- | --- | --- |
| RMSNorm | `test_rmsnorm` | `__shfl_xor_sync` block reduction, fp32 accumulate |
| Rotary embedding | `test_rotary_embedding` | "half" (neox / rotate_half) layout only, head dims 64 and 128 |
| SiluAndMul (SwiGLU) | `test_activation_swiglu` | vectorised `half2`, scalar fallback for an odd width |
| Paged attention + KV cache | `test_paged_attention` | dense 4-D cache, decode scatter and `paged_attention_decode_v1` |
| MatMul | `test_matmul` | cuBLAS `cublasGemmEx`, fp16 storage with fp32 accumulate, `CUBLAS_OP_T` weights |

MatMul is the one operator here with no kernel of its own. A hand-written GEMM
would be measuring the kernel this suite just wrote rather than anything a
deployment runs, and Cutlass is a dependency the suite does not want; cuBLAS
ships with the toolkit and is what a CUDA deployment actually calls for a linear
projection, so it is what the projections are checked against. The wrapper is in
`common/cuda_runtime.hpp` (`CublasGemmFp16`), which is also where the row-major
to column-major argument order is written down.

### Dependencies

The CUDA Runtime API (`<cuda_runtime.h>`, `<cuda_fp16.h>`) and `cudart`, plus
cuBLAS (`<cublas_v2.h>`) for the linear projections. Both ship with the toolkit,
so there is nothing to install beyond CUDA itself. No PyTorch, no Torch-CUDA
headers, no Cutlass, no Python bindings. Every kernel other than the GEMM is
written out by hand in `kernels/cuda/`.

### Prerequisites

- CUDA toolkit 11.0 or newer, with `nvcc` on `PATH`
- CMake >= 3.18
- An NVIDIA GPU for the parity tests; the host-only tests need none

### Configure and build

```bash
cmake -S csrc/tests -B build/csrc-tests-cuda -DENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build/csrc-tests-cuda -j
```

The default architecture list is `75;80;86;89;90`, plus `120` when the toolkit
is 12.8 or newer. Building six architectures is slow and rarely what you want
while iterating, so pass the one you have:

```bash
cmake -S csrc/tests -B build/csrc-tests-cuda -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
```

On Windows, CUDA's MSBuild integration is only installed for Visual Studio
versions the CUDA installer recognises. If configuring reports `No CUDA toolset
found`, use Ninja from a developer command prompt instead, which drives `nvcc`
directly and needs no integration:

```bash
cmake -G Ninja -S csrc/tests -B build/csrc-tests-cuda -DENABLE_CUDA=ON
```

An Ascend build directory next to this one is reused for googletest rather than
downloading a second copy; `-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST` and
`-DVLLM_ASCEND_TESTS_FETCH_GTEST=OFF` work exactly as they do for CANN.

### Run

```bash
ctest --test-dir build/csrc-tests-cuda --output-on-failure
```

```bash
./build/csrc-tests-cuda/test_paged_attention --gtest_filter='*seq4_h32_kv8*'
```

```bash
CUDA_TEST_DEVICE_ID=1 ./build/csrc-tests-cuda/test_rmsnorm
```

`-DVLLM_ASCEND_TESTS_BUILD_BENCHMARKS=ON` is accepted and ignored: the `bench_*`
binaries drive aclnn operators through `common/benchmark.cpp` and have no CUDA
counterpart yet.

### What the kernels commit to

Each one reproduces the arithmetic in `common/cpu_reference.cpp` operation for
operation, not just to within a tolerance:

- **RMSNorm** multiplies in the reference's order, `(x * rstd) * gamma`, and
  computes `1 / sqrtf(...)` rather than `rsqrtf`, whose approximation is visible
  at the `rstd` around 1e3 that a near-zero row produces.
- **SwiGLU** evaluates `v / (1 + expf(-v))`, not `v * sigmoid(v)`, and uses the
  accurate `expf` rather than `__expf`: the saturation case drives the gate to
  +/-30, where the fast intrinsic is not accurate enough.
- **Rotary** reads both halves of the cos/sin row separately even though
  `GatherFullCosSin` duplicates them, so the kernel touches exactly the elements
  the reference touches.
- **Paged attention decode** keeps the reference's phase order and its
  `weight = score * (1/denominator)` factorisation, and accumulates over
  positions in ascending order for each output element.

What is left is reduction association - a tree on the device against a
sequential loop on the host - which is fp32 noise well under the fp16 rounding
of the output. Every comparison uses the same tolerances as the Ascend suite.

### Differences from the Ascend backend

- **The KV cache is the dense 4-D GPU layout**
  `[num_blocks, num_kv_heads, block_size, head_dim]`, not the 310P 5-D fractal
  NZ shape. `reference::PagedKvLayout::kind` selects between them, so
  `ReshapeAndCache` and `PagedAttentionDecode` are still one implementation
  checked by both backends.
- **Rotary covers the "half" layout only.** The interleave layout is exercised
  in the host-only reference tests but has no CUDA kernel.
- **`paged_attention_decode_v1` holds the whole score row in shared memory**,
  the same constraint vLLM's own v1 kernel has. The launcher checks the
  requirement against the device limit and throws a named error rather than
  failing the launch with a bare invalid-value status.
- **The 310P alignment rules do not apply** - the 32-byte SwiGLU gate, the
  fractal width, the 64/128 block sizes. The shape tables still drive both
  backends so the coverage matches, and the tests that assert those rules stay
  on the Ascend side where they mean something.

`test_paged_attention` is worth calling out: the decode half of
`test_paged_attention_310p` is permanently skipped, because CANN 9.1.0 exposes
no aclnn paged attention and `torch_npu._npu_paged_attention` is an ATB C++
object API the aclnn path cannot drive. Until that changes, the CUDA backend is
the only place in the suite where `reference::PagedAttentionDecode` is checked
against a real kernel.

---

## First run on a new CANN release

The suite resolves every aclnn operator with `dlopen`/`dlsym` rather than
linking `libopapi.so` directly, the same way
`csrc/aclnn_torch_adapter/op_api_common.h` does. A renamed or absent operator
becomes a skip naming the symbol instead of a link error that takes out the
whole binary.

The trade-off is that the argument lists in `common/aclnn_ops.hpp` are declared
by hand and are **not** checked by the compiler. Before trusting the first run
on a CANN version this suite has not seen, confirm each prototype:

```bash
grep -rA24 'GetWorkspaceSize' $ASCEND_HOME_PATH/include/aclnnop/aclnn_rms_norm.h $ASCEND_HOME_PATH/include/aclnnop/aclnn_swi_glu.h
```

```bash
grep -rlA24 'ApplyRotaryPosEmb\|ReshapeAndCache\|PagedAttention' $ASCEND_HOME_PATH/include/aclnnop/
```

A mismatch shows up as a non-zero status from the planning call with the CANN
diagnostic attached, not as silent corruption. `aclnnPagedAttention` carries the
most optional arguments of the five and is the most likely to differ.

The inventory printed at start-up tells you what resolved:

```
[ascend-test] aclnn operator inventory
[ascend-test]   aclnnRmsNorm                 found
[ascend-test]   aclnnSwiGlu                  found
[ascend-test]   aclnnApplyRotaryPosEmbV2     found
...
```

---

## Profiling with msprof

Collect a full trace of one suite:

```bash
msprof --application="./build/csrc-tests/test_rmsnorm_310p --gtest_filter=*tokens128_hidden8192*" --output=./prof/rmsnorm --aic-metrics=PipeUtilization --ai-core=on
```

Narrow the filter to a single parameterised case. Without it the trace covers
every shape in the suite and the per-kernel timings are hard to attribute.

Task-level timeline only, which is much cheaper:

```bash
msprof --application="./build/csrc-tests/test_paged_attention_310p --gtest_filter=*seq4_h32_kv8_d128_b64_ctx512*" --output=./prof/pa --task-time=on --aicpu=on
```

Export the collected data to CSV:

```bash
msprof --export=on --output=./prof/rmsnorm
```

Then read `./prof/rmsnorm/**/mindstudio_profiler_output/op_summary_*.csv` for
per-operator duration, and `op_statistic_*.csv` for the aggregate.

Two things to keep in mind when profiling a **test** binary:

- Every `RunAclnn` call synchronises the stream, so the timings include launch
  overhead and are not a throughput measurement.
- The workspace is allocated and freed per call. In the plugin it comes from the
  caching allocator, so a profile taken here overstates allocation cost.

Neither applies to the `bench_*` binaries, which is what they exist for. Profile
those instead when the question is cost rather than correctness, and narrow the
run with `ASCEND_BENCH_MODES=pipelined` so the trace covers one timing strategy:

```bash
ASCEND_BENCH_MODES=pipelined msprof --application="./build/csrc-tests/bench_rmsnorm_310p" --output=./prof/rmsnorm --aic-metrics=PipeUtilization --ai-core=on
```

For a memory-access view instead of a pipe-utilisation view:

```bash
msprof --application="./build/csrc-tests/test_activation_swiglu_310p" --output=./prof/swiglu --aic-metrics=MemoryUB
```

---

## 310P specifics encoded in the tests

These come from the plugin source rather than from general Ascend documentation,
and several differ from the GPU defaults:

- **KV cache is 5-D NZ**, not the generic `[2, num_blocks, block_size,
  num_kv_heads, head_size]`. `AscendAttentionBackend310.get_kv_cache_shape`
  returns `(2, num_blocks, num_kv_heads * head_size / 16, block_size, 16)` and
  the runner allocates each half with `acl_format=ACL_FORMAT_FRACTAL_NZ`. This
  requires `num_kv_heads * head_size % 16 == 0`.
- **Block size is 64 or 128**, not 16 or 32.
  `get_supported_kernel_block_sizes()` returns `[128, 64]`, and the runner
  additionally requires `block_size * head_size <= 128 * 128`
  (`_ATTENTION_BLOCK_SIZE_LIMIT`).
- **SwiGLU needs `x.shape[-1] % 32 == 0`**. `AscendSiluAndMul310.forward` falls
  back to eager torch otherwise, so a width that fails the gate never reaches
  the kernel.
- **Rotary embedding supports head dims 64 and 128 only**, per the
  `self.rotary_dim in (64, 128)` gate in the 310P rotary module.
- **Allocations are padded to 32 bytes.** MTE2/MTE3 move 32-byte bursts, so an
  unpadded tail lets the last burst run past the end of the buffer.

## Scope

Covered: numerical parity against fp32 CPU references, layout and alignment
contracts, GQA head mapping, both rotary layouts, block-table paging across
multiple blocks, context-length bounds, and per-shape latency and throughput.

Not covered: the quantised (W8A8 / int8 KV cache) paths, chunked prefill and the
splitfuse attention variants, multi-device or graph-capture execution, and
performance regression *thresholds* - the benchmarks report numbers and check
that the operator still produces its output, but nothing fails on a slowdown.
On the CUDA backend, additionally: MatMul, the interleave rotary layout, and the
benchmarks.
The five kernels here are the fp16 decode path only, which is what the brief
scoped. [COVERAGE.md](COVERAGE.md) has the full gap list.

## Adding a kernel

1. Declare the operator in `common/aclnn_ops.hpp` with a `mirrors:` comment
   naming the `torch_npu` entry point and the vllm-ascend call site, and add it
   to `ProbeAllOperators`.
2. Add a reference to `common/cpu_reference.{hpp,cpp}`.
3. Add `kernels/test_<name>_310p.cpp` with host-only reference tests plus the
   device parity tests.
4. Append the binary name to `VLLM_ASCEND_KERNEL_TESTS` in `CMakeLists.txt`.

To add a benchmark for it, write `kernels/bench_<name>_310p.cpp` defining
`bench::kSuiteName` and `bench::BuildSuite`, and append the binary name to
`VLLM_ASCEND_KERNEL_BENCHMARKS`. Allocate everything in `BuildSuite`, hand
`PlanAclnn` the same arguments `RunAclnn` would take, and give the case a
checksum.

For the CUDA backend:

1. Reuse the reference from step 2 - do not write a second one. If the kernel
   needs a different memory layout, add it as a variant behind the existing
   struct the way `PagedKvCacheLayout` does, so both backends stay checked
   against one implementation.
2. Declare a launcher in `kernels/cuda/cuda_kernels.hpp`, passing fp16 buffers
   as `uint16_t*`. That header is the seam between the host compiler and nvcc,
   so it must name no CUDA language extension and no `__half`.
3. Write `kernels/cuda/<name>_kernel.cu` and add it to
   `vllm_ascend_cuda_kernels` in `CMakeLists.txt`.
4. Add `kernels/cuda/test_<name>.cpp` and append the binary name to
   `VLLM_ASCEND_CUDA_KERNEL_TESTS`.
