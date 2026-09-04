# Bare-metal kernel tests and benchmarks (Ascend 310P3)

A standalone suite for the five operator families a **Qwen3.5** forward pass
needs, driven straight through the CANN runtime. No Python, no PyTorch, no
`torch_npu`, no Torch C++ ABI.

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
│   ├── cpu_reference.hpp / .cpp         naive fp32 references for all five kernels
│   ├── device_buffer.hpp                RAII device allocation, 32-byte default, 512 for benchmarks
│   ├── device_tensor.hpp                device buffer + aclTensor descriptor, with host conversions
│   ├── fp16.hpp                         IEEE-754 binary16 conversion, round-to-nearest-even
│   ├── main.cpp                         entry point for the test_* binaries, prints the inventory
│   ├── qwen_shapes.hpp                  Qwen3.5 shapes and the 310P alignment rules
│   ├── random_data.hpp                  deterministic, platform-independent test data
│   ├── tensor_compare.hpp               allclose with a diagnostic report
│   ├── test_benchmark_harness.cpp       device-free tests over the benchmark report arithmetic
│   └── test_harness.hpp / .cpp          AscendTestEnvironment: aclInit, device, context, stream
└── kernels/
    ├── test_matmul_310p.cpp             bench_matmul_310p.cpp
    ├── test_rmsnorm_310p.cpp            bench_rmsnorm_310p.cpp
    ├── test_rotary_embedding_310p.cpp   bench_rotary_embedding_310p.cpp
    ├── test_activation_swiglu_310p.cpp  bench_activation_swiglu_310p.cpp
    └── test_paged_attention_310p.cpp    bench_paged_attention_310p.cpp
```

Each test file has two layers. Tests named `*Reference` and `*Shapes` are
host-only: they check the CPU reference and the layout arithmetic and run
anywhere, including on a build machine with no NPU. The rest need a device and
skip with an explanatory message when one is not attached.

---

## Prerequisites

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

`test_benchmark_harness` is the exception to all of the above: it covers the
benchmark harness rather than a kernel, needs no device, and passes on the build
host. It is the fastest check that a change to `common/benchmark.*` or
`common/device_buffer.hpp` did not move what a report says.

```bash
./build/csrc-tests/test_benchmark_harness
```

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
  workspace first-touch and AI Core clock ramp do not land in sample 0, and a
  hard `aclrtSynchronizeStream` separates it from the timed loop: every warmup
  launch and every MTE transfer it queued has retired before sample 0 starts.

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

### Samples that are not durations

The `pipelined` and `device` rows come from `aclrtEventElapsedTime`, which
reports milliseconds as a `float` and has several ways of returning something
that is not a duration at all: an event the device has not resolved, an event
created without a timestamp, a wrapped device counter. Any of those come back
negative or non-finite, and one of them in a set is enough to drag the mean
below zero and make the derived TFLOP/s and GB/s meaningless.

So every sample is validated before it is aggregated:

- Events are created with `ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC` where the
  runtime accepts it, so they carry a timestamp and are host-waitable. The
  report header names which flag combination the run actually got; a run that
  fell back to `aclrtCreateEvent` says so, and its device rows are suspect.
- Both events are resolved, and the whole stream drained, before any elapsed
  time is queried.
- A sample that is not positive and finite is dropped, not clamped or replaced.
  The report lists what was dropped, per case and per mode, under
  `discarded samples`, and the CSV carries it in a `discarded` column next to
  `samples`. A median over 40 of 100 samples is a different claim from a median
  over all 100, so the run says which it is.
- A mode with **no** usable sample fails its case. A row of zeros reads like a
  fast kernel, so the suite refuses to print one.

The `host` row is taken from `steady_clock` in the clock's own 64-bit nanosecond
representation and converted once, in `double`. Nothing on that path narrows:
microseconds in a 32-bit integer wrap after 2.14 seconds, which is inside the
range a large prefill matmul reaches.

### Knobs

Every `ASCEND_BENCH_*` integer is parsed with a range. A value that is not an
integer, or is outside the range, is reported on stderr and the default is used
instead -- a typo in `ASCEND_BENCH_WARMUP` no longer silently becomes zero
warmup.

| Variable | Default | Effect |
| --- | --- | --- |
| `ASCEND_BENCH_WARMUP` | 20 | warmup iterations, 0 to 1000000 |
| `ASCEND_BENCH_ITERS` | 100 | timed iterations per mode, 1 to 1000000 |
| `ASCEND_BENCH_BATCH` | 10 | launches per event pair in `pipelined`, 1 to 766: two events bracket a batch, and a deeper one than the stream holds would be split by the runtime's back-pressure rather than measuring a fuller pipeline |
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
