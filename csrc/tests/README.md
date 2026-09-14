# Bare-metal kernel tests and benchmarks (Ascend 310P3 and 950PR)

A standalone suite for the five operator families a **Qwen3.5** forward pass
needs, driven straight through the CANN runtime. No Python, no PyTorch, no
`torch_npu`, no Torch C++ ABI.

The suite is split into three **execution tiers** - host, simulator and device -
plus the Ascend 310P leg, which is a different part rather than a tier. Which of
them a build produces is decided by `SOC_VERSION` and `RUN_MODE`; see
[Layout](#layout). A 950PR build contains no 310P target at all.

Ascend 310P3 is the default `SOC_VERSION` and is what everything up to
[Ascend 950PR](#ascend-950pr) describes.
[TURBOQUANT_TESTS.md](TURBOQUANT_TESTS.md) reviews the 950PR tiers case by case.

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

The suite is organised by **where a test can run**. Three tiers, mutually
exclusive by construction, plus the 310P leg, which is a different part rather
than a tier.

```
csrc/tests/
|-- CMakeLists.txt          tier dispatch, SoC gating, the shared helpers
|-- COVERAGE.md             C++ vs Python coverage and parity audit
|-- TURBOQUANT_TESTS.md     the tier map and every TurboQuant case, reviewed
|
|-- common/                 shared infrastructure (see the table below)
|-- reference/              turbo_quant_cpu.h, the CPU oracle
|-- data/golden_layer3/     Git LFS: weights, taps and output of one Qwen3.5 layer
|-- turboquant/             ascendc_library() for the Ascend C kernels
|
|-- host/                   TIER 1 -- no CANN at all, no NPU, runs anywhere
|   `-- test_host_turboquant_fidelity.cpp
|
|-- sim/                    TIER 2 -- CAModel only; links libruntime_camodel.so
|   |-- test_sim_950pr_turboquant_kernels.cpp
|   |-- test_sim_950pr_turboquant_decode.cpp
|   |-- test_sim_950pr_cube_hadamard.cpp          spike: one shape of the sweep below
|   `-- sim_hadamard_hybrid_kernels.cpp           spike: test-owned Ascend C, not the decode
|
|-- device/                 TIER 3 -- physical 950PR silicon; every timing
|   |-- test_device_950pr_turboquant.cpp          production shapes, camodel refused
|   |-- test_device_950pr_matmul.cpp
|   |-- test_device_950pr_rmsnorm.cpp
|   |-- test_device_950pr_rotary_embedding.cpp
|   |-- test_device_950pr_activation_swiglu.cpp
|   |-- test_device_950pr_qwen_layer_golden.cpp
|   |-- test_device_950pr_benchmark_harness.cpp   needs no device; links acl.h
|   |-- test_device_950pr_cube_hadamard.cpp       spike: D x V sweep against the CPU golden
|   |-- bench_950pr_cube_hadamard.cpp             spike: the same sweep, timed, --csv=
|   |-- bench_main_950pr_hadamard.cpp             its entry point; only it takes argv
|   |-- bench_device_950pr_turboquant.cpp         the AIV-only baseline
|   `-- bench_device_950pr_turboquant_ablation.cpp  the kv4fp8 Cube split, cut stage by stage
|
`-- device_310p/            the 310P leg -- NOT configured under a 950PR SoC
    |-- test_*_310p.cpp
    `-- bench_*_310p.cpp
```

| `common/` | |
| --- | --- |
| `acl_check.hpp` | `ACL_CHECK` / `ASSERT_ACL_OK`, with `aclGetRecentErrMsg` attached |
| `aclnn_ops.hpp` / `.cpp` | the version-sensitive aclnn prototypes -- read this first. Shared by both parts, despite the history |
| `aclnn_ops_950pr.hpp` / `.cpp` | the 950PR operator audit and its extra prototypes, including FIA V5 |
| `aclnn_runtime.hpp` / `.cpp` | dlopen/dlsym loader, aclTensor RAII, two-phase launch |
| `ascend950_shapes.hpp` | Qwen3.5-2B layer 3 and the arch35 platform rules |
| `bench_main.cpp` / `bench_main_950pr.cpp` | benchmark entry points; the 950PR one names the part and refuses a camodel |
| `benchmark.hpp` / `.cpp` | plan-once launch, event timing, statistics, reporting |
| `cpu_reference.hpp` / `.cpp` | naive fp32 references for all five kernels |
| `device_buffer.hpp` | RAII device allocation, 32-byte default, 512 for benchmarks |
| `device_tensor.hpp` | device buffer + aclTensor descriptor, with host conversions |
| `fp16.hpp` | IEEE-754 binary16 conversion, round-to-nearest-even |
| `golden_layer3.hpp` / `.cpp` | LFS-aware loader for the layer-3 dump |
| `hadamard_spike.hpp` | the Cube-Hadamard spike's constant images, chunk planning and launchers. Exploratory; not on the decode path |
| `main.cpp` / `main_950pr.cpp` | test entry points; the 950PR one prints its tier and whether a camodel is loaded |
| `partial_rotary_950pr.hpp` / `.cpp` | partial RoPE: custom operator, else packed stock operator |
| `qwen_shapes.hpp` | Qwen3.5 shapes and the 310P alignment rules |
| `random_data.hpp` | deterministic, platform-independent test data |
| `tensor_compare.hpp` | allclose with a diagnostic report |
| `test_harness.hpp` / `.cpp` | `AscendTestEnvironment`, and the camodel detection the device tier gates on |
| `turboquant_launch.hpp` / `.cpp` | torch-free binding for the TurboQuant kernels: the two `_impl` prototypes, the codec table image, the grid maths |

### Which tiers a configuration builds

`RUN_MODE` selects between the two device-side tiers, and it has to:
`ascendc_library()` builds one kernel library per `RUN_MODE`, so a tree cannot
hold both a camodel and a silicon build of it. That is what makes the device
tier's "no CAModel fallback" structural rather than aspirational.

| Configuration | Tiers |
| --- | --- |
| `-DVLLM_ASCEND_TESTS_HOST_ONLY=ON` | `host/` |
| `-DSOC_VERSION=Ascend950PR_9599` (RUN_MODE defaults to `npu`) | `host/` + `device/` |
| `-DSOC_VERSION=Ascend950PR_9599 -DRUN_MODE=sim` | `host/` + `sim/` |
| `-DSOC_VERSION=Ascend310P3` | `host/` + `device_310p/` |

**A 950PR build excludes the 310P targets entirely** -- `device_310p/` is never
configured, so no `*310*` binary exists in the tree and `ctest -N` has none to
list. Configure output says so:

```
-- vllm-ascend tests: tiers -> host=TRUE sim=FALSE device=TRUE device_310p=FALSE
-- vllm-ascend tests: 310P targets are excluded from this build entirely
```

Every test carries its tier as a ctest label, so the tiers are selectable
without knowing the binary names:

```bash
ctest -L host                 # tier 1
ctest -L device -LE benchmark # tier 3 correctness
ctest -L benchmark            # tier 3 timings
```

Within a device-side binary, tests named `*Reference`, `*Shapes` and
`*LaunchContract` still run with no NPU attached: they check the CPU reference
and the layout arithmetic. The rest skip with an explanatory message when no
device is present.

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

A benchmark exits 77 when no usable device of the part it targets is attached,
which ctest reports as a skip rather than a failure.
`-DVLLM_ASCEND_TESTS_BUILD_BENCHMARKS=OFF` leaves them out of the build
entirely.

Which part that is comes from the entry point: `common/bench_main.cpp` for the
five 310P suites, `common/bench_main_950pr.cpp` for `bench_device_950pr_turboquant`,
which appears only in a 950PR device-tier build alongside the TurboQuant
kernels. It is the one benchmark in the suite that times kernels built here
rather than a stock aclnn operator, and the only one that reports an analytic
traffic model next to the measurement -
[TURBOQUANT_TESTS.md](TURBOQUANT_TESTS.md) §8 explains why it has to.

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
| `ASCEND_BENCH_WARMUP` | 20 | warmup iterations, 0 to 1000000. A case that defines a checksum is floored at 1: the reference checksum is read straight after the warmup, and at 0 it would be taken over the allocation's zero fill and fail the case at the end of the timed loop |
| `ASCEND_BENCH_ITERS` | 100 | timed iterations per mode, 1 to 1000000 |
| `ASCEND_BENCH_BATCH` | 10 | launches per event pair in `pipelined`, 1 to 766: two events bracket a batch, and a deeper one than the stream holds would be split by the runtime's back-pressure rather than measuring a fuller pipeline |
| `ASCEND_BENCH_MODES` | all three | comma-separated subset of `pipelined,device,host` |
| `ASCEND_BENCH_CSV` | unset | write one row per (case, mode) to this path |
| `ASCEND_BENCH_REPEATABLE` | on | `0` forces the re-plan-per-launch path |
| `ASCEND_BENCH_TQ_CONTEXTS` | `512,1024,2048` | `bench_device_950pr_turboquant` only: the context lengths to sweep |
| `ASCEND_BENCH_TQ_ABLATION_DIMS` | `256,512` | `bench_device_950pr_turboquant_ablation` only: head sizes, powers of two in [64, 512] |
| `ASCEND_BENCH_TQ_ABLATION_CONTEXTS` | `64,512,1024,2048` | `bench_device_950pr_turboquant_ablation` only: contexts, positive multiples of 8 (TURBOQUANT_TESTS.md 13.20) |
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
- `bench_device_950pr_turboquant` sweeps context 512 / 1024 / 2048 at Qwen3.5-2B's
  `head_dim` 256, timing the 4-bit cache write and the AIV-only split/combine
  decode, with an fp16 decode through `aclnnFusedInferAttentionScoreV2` as the
  baseline where that operator exists.
- `bench_device_950pr_turboquant_ablation` times the kv4fp8 Cube split cut at
  each `DecodeAblationStage` -- MTE2 read, unpack, query rotation, L1 staging,
  score GEMM, full pipeline -- over D in {256, 512} and S in {64, 512, 1024,
  2048}, and prints a per-stage latency waterfall. See TURBOQUANT_TESTS.md 7.6.

See [COVERAGE.md](COVERAGE.md) for what this does and does not close, and
[TURBOQUANT_TESTS.md](TURBOQUANT_TESTS.md) for the TurboQuant leg specifically.

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

---

## Ascend 950PR

A second, opt-in leg of the same suite targets the **Ascend 950PR** (A5, DaVinci
arch35). It shares the CPU references, the fp16 conversion, the deterministic
data generator and the tolerance machinery with the 310P leg; what differs is
the part, the operators, and one shape the 310P cannot run at all.

```bash
cmake -S csrc/tests -B build/csrc-tests-950pr -G Ninja \
      -DSOC_VERSION=Ascend950PR_9599
```

`SOC_VERSION` is the only switch: naming a 950PR bin selects the 950PR tiers and
**excludes the 310P targets entirely** - they are not configured, not compiled,
and contribute no target. The old `-DENABLE_ASCEND_950PR=ON` flag is gone, and a
configure that still passes it with a non-950 `SOC_VERSION` fails with a message
saying so rather than silently building the wrong leg.

| Binary | Stage | Operator |
| --- | --- | --- |
| `test_device_950pr_matmul` | 2, 6, 8 — every linear projection | `aclnnMatmul` (cube) |
| `test_device_950pr_rmsnorm` | 1, 7 | `aclnnRmsNorm` |
| `test_device_950pr_rotary_embedding` | 3, 4 — **partial** RoPE | `aclnnInplacePartialRotaryMul`, else packed `aclnnApplyRotaryPosEmbV2` |
| `test_device_950pr_activation_swiglu` | 8 | `aclnnSwiGlu` |
| `test_device_950pr_qwen_layer_golden` | 1–9 end to end | all of the above plus `aclnnScatterPaKvCache`, `aclnnFusedInferAttentionScoreV2`, `aclnnSigmoid`, `aclnnMul`, `aclnnInplaceAdd` |
| `test_sim_950pr_turboquant_kernels` | 5 — decode, 4-bit KV cache | the TurboQuant kernels out of `csrc/attention/turboquant`, not an aclnn operator. **Sim tier** |
| `test_sim_950pr_turboquant_decode` | 5 — one decode pass end to end | ditto, with `aclnnFusedInferAttentionScore*` as an optional unquantised control. **Sim tier** |
| `test_device_950pr_turboquant` | 5 — the same, at Qwen3.5-2B's real shapes | ditto. **Device tier**: silicon only, refuses a camodel |
| `bench_device_950pr_turboquant` | 5 — the AIV-only decode, timed | ditto, with `aclnnFusedInferAttentionScoreV5` (V2 fallback) as the fp16 baseline |
| `bench_device_950pr_turboquant_ablation` | 5 — the kv4fp8 Cube split, timed stage by stage | the TurboQuant kernels, plus five test-only entry points built under `VLLM_ASCEND_TQ_DECODE_ABLATION` |

`common/aclnn_ops_950pr.hpp` carries the operator audit: which stage runs on a
stock CANN operator, which on a kernel built out of `csrc/`, and the CANN header
each prototype was verified against.

### The TurboQuant leg, and running it on the simulator

> **[TURBOQUANT_TESTS.md](TURBOQUANT_TESTS.md) is the full review of this leg**:
> every case in all five TurboQuant binaries, what it validates, what its bound
> is and where that number came from, plus the coverage gaps. What follows here
> is the build-and-run summary only.

The last four binaries in that table are unlike everything else in this suite:
they drive kernels that are **compiled here**, from
`csrc/attention/turboquant/turboquant_kernels.cpp` - the same source the wheel
builds, not a copy - rather than calling an operator CANN already shipped. That
makes them the only part of the project that needs the Ascend C kernel
toolchain (`ccec` / `bisheng` and `tools/tikcpp/ascendc_kernel_cmake`), which is
why they sit behind their own option:

```bash
-DVLLM_ASCEND_TESTS_BUILD_TURBOQUANT_KERNELS=OFF   # default ON
```

Because the kernels are built rather than loaded, they can also be *run* with no
950PR attached, on the CANN camodel:

```bash
cmake -S csrc/tests -B build/csrc-tests-950pr-sim -G "Unix Makefiles" \
      -DSOC_VERSION=Ascend950PR_9599 -DRUN_MODE=sim
cmake --build build/csrc-tests-950pr-sim -j

export LD_LIBRARY_PATH=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib:$LD_LIBRARY_PATH
./build/csrc-tests-950pr-sim/test_sim_950pr_turboquant_decode
```

`RUN_MODE=sim` links `libruntime_camodel.so` in place of `libruntime.so`; the
`LD_LIBRARY_PATH` above is what makes the loader pick it up at run time, and
`aclrtGetSocName()` then reports a real `Ascend950PR_*` bin, so the tests run
rather than skip. Three things are worth knowing before you try it:

- **It is cycle-level slow.** One decode pass is two to four *minutes*. That is
  the whole reason `test_sim_950pr_turboquant_decode` runs a single decode step at
  the smallest shape that still exercises the tiling, the paging and the GQA
  mapping, instead of a sweep. ctest gets a 5400 s timeout for these two in
  `sim` mode.
- **Use the Makefiles generator, or expect a bare Python `KeyError`.**
  `ascendc_library()` configures four nested ExternalProjects with the outer
  project's generator, and with Ninja the object path CANN 9.1.0's
  `extract_host_stub.py` looks up in `compile_commands.json` differs from the
  one it is handed by a `/./`. The sub-builds are pinned to `Unix Makefiles`
  for that reason - see `kernels/ascend/turboquant/CMakeLists.txt`.
- **`aclnnFusedInferAttentionScore` V1 to V4 do not exist on an Ascend950.** The
  simulator test uses it as an unquantised fp16 control and reports the refusal
  rather than failing; the exact fp32 host path is what its assertions compare
  against. `bench_device_950pr_turboquant` hits the same wall for its fp16 baseline and
  registers a ctest-visible skip with the reason.
  `common/aclnn_ops_950pr.hpp` records the measurement.

The same binaries run unchanged on silicon: configure without `RUN_MODE=sim`
(or with `-DRUN_MODE=npu`) and they link the real runtime.

**Two of them are for silicon specifically.**
`test_device_950pr_turboquant` sweeps the shapes Qwen3.5-2B actually decodes
at - `head_dim` 256, `block_size` 128, context 64 / 512 / 1024 / 2048 - which is
days of camodel time, so it refuses to run there. The SoC name cannot tell the
camodel from the part, so `REQUIRE_PHYSICAL_ASCEND_950PR` looks for
`libruntime_camodel.so` in `/proc/self/maps` instead and skips with the path it
found. `ASCEND_TEST_ALLOW_SIMULATOR=1` overrides that, and
`ASCEND_TQ_BARE_METAL_CONTEXTS` shrinks the sweep, both for smoke-checking the
binary rather than for producing results.
`test_device_950pr_turboquant_multimode` is the Cube-native decode's validation
matrix on the same part - batch `{1, 8}` by context `{64, 512, 1024, 2048}` at the
production head count, gated on `cos > 0.90` per shape - and compiles the sim
tier's source so the two tiers cannot enforce different bounds;
`ASCEND_TQ_SIM_BATCH` and `ASCEND_TQ_SIM_CONTEXT` narrow either axis. `bench_device_950pr_turboquant` is built
under `RUN_MODE=sim` but disabled in ctest for the same reason;
`ASCEND_BENCH_TQ_CONTEXTS` and the usual `ASCEND_BENCH_*` knobs make a hand-run
smoke check tractable.

`test_host_turboquant_fidelity` needs none of this. It is host-only, links neither
`libascendcl.so` nor a kernel, and reports the codec's fidelity on the Qwen3.5
layer-3 dump from the CPU reference alone - see `-DVLLM_ASCEND_TESTS_HOST_ONLY=ON`.

### Partial rotary is the interesting stage

Qwen3.5 sets `partial_rotary_factor` 0.25 against `head_dim` 256, so channels
`[0, 64)` of every head rotate and `[64, 256)` must come out bit-identical. The
310P cannot run that shape at all (`AscendMRotaryEmbedding310` gates on
`rotary_dim in (64, 128)`), and `aclnnApplyRotaryPosEmbV2` cannot express it
either — it rotates the whole trailing dimension.

`common/partial_rotary_950pr.hpp` resolves that at run time, preferring
`aclnnInplacePartialRotaryMul` — the vllm-ascend custom operator from
`csrc/attention/inplace_partial_rotary_mul`, which has an `ascend950` AICore
config — and falling back to packing the rotary slice of every head into a
contiguous buffer, rotating that with the stock operator, and unpacking. Every
rotary test records which path ran, because the two are different claims about
the same hardware.

The custom operator is **not part of CANN**: it only resolves once the
vllm-ascend custom op package is installed, and it lives in `libcust_opapi.so`
under `$ASCEND_CUSTOM_OPP_PATH` or an `$ASCEND_OPP_PATH/vendors` entry rather
than in `libopapi.so`. `common/aclnn_runtime.cpp` searches those first, in the
same order `csrc/aclnn_torch_adapter/op_api_common.h` does, so a symbol resolves
here to the implementation `torch_npu` would have called.

### Golden layer-3 parity

`test_device_950pr_qwen_layer_golden` runs a whole Qwen3.5 decoder layer on the NPU and
compares seven intermediate taps plus the final output against
`csrc/tests/data/golden_layer3`, produced by `scripts/dump_qwen35_layer3.py`.
The dump is tracked with Git LFS; a tree where it was never fetched gets a skip
naming `git lfs pull` rather than a size mismatch. `QWEN_GOLDEN_LAYER3_DIR`
overrides the baked-in location.

That binary also carries **13 host-only tests that need no NPU at all**: the
same nine stages run again in float on the host, rounding to fp16 at every stage
boundary the way the dumper does, plus the loader's failure modes. They check
that the dump is internally consistent, that the loader reads it correctly and
that the CPU references agree with PyTorch — on a build machine.

```bash
./test_device_950pr_qwen_layer_golden --gtest_filter='QwenLayer3DumpTest.*:QwenLayer3Loader.*'
```

### What the shipped dump does not test

It is `pos=0`, `ctx_len=1`. At position 0 the rotary tables are `cos=1, sin=0`,
so RoPE is the identity. With one context position the softmax is exactly 1.0,
so the attention context is just V and **the layer output does not depend on Q,
K or RoPE**. The taps are what catch a bug in those stages, which is why they
exist; `Stage5DecodeAttentionContextIsTheCachedValue` turns the softmax identity
into a direct check on the paged KV write, the block table and the GQA mapping.

`--pos N` regenerates a set where the rotary stages carry weight. `--ctx-len M`
for `M > 1` does **not** work: the dumper generates the prior context but never
writes it out, so the test has no way to reconstruct it. Teaching the dumper to
write `k_past` / `v_past` is a prerequisite for any multi-position dump.

---

## Scope

Covered: numerical parity against fp32 CPU references, layout and alignment
contracts, GQA head mapping, both rotary layouts, block-table paging across
multiple blocks, context-length bounds, and per-shape latency and throughput.

Also covered, since the TurboQuant leg landed: the 4-bit rotated KV cache - its
write path and its split/combine decode, bin for bin against the CPU reference,
plus the fidelity of the scheme itself against exact fp32 attention.  The write
path is compared in bin indices rather than bytes because the codec's RMS scale
is a sum the device reduces in a tree and the host sums serially: the two agree
to within a coordinate landing either side of one decision boundary, and the
test bounds exactly that.

Not covered: the W8A8 and int8 KV cache paths, chunked prefill and the splitfuse
attention variants, multi-device or graph-capture execution, and performance
regression *thresholds* - the benchmarks report numbers and check that the
operator still produces its output, but nothing fails on a slowdown. Everything
outside the TurboQuant binaries is the fp16 decode path only, which is what the
brief scoped. [COVERAGE.md](COVERAGE.md) has the full gap list.

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
