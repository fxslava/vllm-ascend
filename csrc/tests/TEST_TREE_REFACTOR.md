# Hardware test layout and shared DSV4 abstractions

```text
csrc/tests/
  common/          oracle, metrics, deterministic data, device owners, timing
  host/            runtime-free arithmetic and abstraction tests
  sim/             CAModel harnesses
  device_950pr/    950PR correctness, CANN V5 benchmarks, profiling
  device_910b/     910B correctness; comparative benchmark reserved for validation
  device_310p/     existing 310P correctness and benchmarks, unchanged target names
  ascendc/         one CANN compiler integration for all kernel libraries
  reference/      TurboQuant reference and production tiling data
```

The deprecated `device/` directory and the old DSV4 reference header are removed.
The oracle now lives only in `common/dsv4_test_oracle.hpp`. Existing independent
FP64 expectations in host tests remain independent of that oracle.

## Audit and shared responsibilities

| Common abstraction | Responsibility |
| --- | --- |
| `dsv4_test_oracle.hpp` | E2M1/E8M0 decoding, ascending FP32 accumulation, symmetric `std::clamp`, `up * swish(gate)`, BF16 round-to-even/widening, geometry and explicit UB payload |
| `dsv4_metrics.hpp` | maximum finite BF16 ULP distance, strictly-over-tolerance mismatch rate/count, separate NaN/signed-infinity classification mismatches |
| `dsv4_synthetic_data.hpp` | deterministic integer generator, exhaustive packed bytes, configurable E8M0 range, all-byte scale sweep and both saturation signs |
| `ascend_device_context.hpp` | existing device/HBM owners, shared stream and pinned staging owners, synchronized asynchronous H2D/D2H transfers |
| `ascend_concurrent_runner.hpp` | validate distinct streams and enqueue all experts before reading any output; independently tested without CANN |
| `ascend_benchmark_runner.hpp` | 50 warmups and 200 device-event samples through the existing benchmark event pool/statistics; median latency, effective TFLOPS and logical bandwidth |
| `dsv4_device_case.hpp` | expert-specific HBM ownership, kernel launch and output retrieval; optional borrowed stream preserves simulator stream-order tests |
| `dsv4_production_suite.hpp` / `dsv4_test_checks.hpp` | one six-case production suite and one set of assertions, instantiated by both hardware wrappers |
| `dsv4_benchmark_suite.cpp` | shared packed-FP4 versus pre-dequantized CANN GroupedMatmulV5 pipeline and pre/post-timing numerical checks |

The 310P tests and other 950PR operator tests already use common device tensors,
reference functions, assertion helpers and the generic benchmark runner. Their
operator-specific setup is retained; another parallel allocator or timer was not
introduced. The existing `AscendDevice` and `DeviceBuffer` implementations remain
the sole owners of device initialization/reset and aligned HBM allocation/free.

The deterministic production generator preserves its existing default data.
The simulator's former standard-library normal-distribution generator now uses
the same reproducible integer generator, so simulator input bits change while
its golden comparison and single-stream ordering checks remain intact.

One-shot ACLNN preparation runs outside each event interval, with synchronization
before the next prepared sample. Pipelined timing rejects prepared one-shot plans
before any callback, avoiding executor/workspace reuse across outstanding samples.
Two regression cases cover this mode validation in the benchmark harness.

## Duplication count

At the start of this refactor the two production test sources totaled 337 lines.
They now total 38 lines; the shared suite and assertion helper add 188 lines.
Thus the consolidated production-suite implementation is **226 lines, 111 fewer**
than the two original implementations, while both parts now run all six cases.
This is a scoped net line reduction, not a claim that every removed line was an
identical duplicate. The benchmark wrapper shrank from 308 to 32 lines; most of
that code moved into the shared pipeline and is not counted as eliminated code.

## Target matrix

All configurations include four runtime-free host executables.

| SoC / mode | Additional targets | Total executables |
| --- | --- | ---: |
| Ascend950PR_9599 / npu | `device_950pr`: DSV4, stock operators, TurboQuant, benchmarks, profiler | 18 |
| Ascend910B1..B4 / npu | `device_910b`: `test_device_910b_dsv4_moe_expert` | 5 |
| Ascend310P* / npu | `device_310p`: five correctness and five benchmark binaries | 14 |
| Ascend950PR_9599 / sim | `sim`: ten CAModel/compile-check binaries; physical target names excluded | 14 |
| HOST_ONLY=ON | host only; no CANN discovery or linking | 4 |

The 910B leg excludes TurboQuant arch35 probes and 310P targets. Unknown SoCs
fail configuration instead of being routed silently to 310P. The existing 950PR
simulator leg remains buildable without physical hardware. A 310P sim request
configures host targets only; a 910B sim request is explicitly unsupported.
DSV4 and TurboQuant kernel options are independent, and every Ascend C library
has explicit `--cce-auto-sync=off` and `-Werror` flags.

The 910B comparative benchmark remains planned: it can reuse the common pipeline
after GroupedMatmulV5 BF16-to-FP32 tuples are qualified on that device. No timings
or unsupported baseline fallback are advertised.

## Verification

Validation used WSL, GCC 11.4 and CANN 9.2.0-beta.2, with
`VLLM_ASCEND_TESTS_WERROR=ON`. Device link checks used CANN development driver
stubs because this workstation exposes no physical NPU.

- Complete 950PR and 910B device configurations compiled from fresh build trees.
- Complete 310P device and 950PR simulator configurations compiled.
- All four host CTest entries passed; original DSV4 oracle cases remain 20/20,
  and ten new common-abstraction cases passed.
- `readelf` confirms the host abstraction test depends only on ordinary C++/C
  system libraries. Device RUNPATHs exclude CAModel directories.
- Target manifests confirm hardware isolation and the counts above.
- `clang-format --dry-run -Werror` and `git diff --check` passed for the
  refactored C++ sources and headers.

These are compile and host checks. NPU accuracy, concurrency, event timing and
the benchmark preparation path still require silicon execution. The prior
device launch failed before tests with a CANN runtime symbol error; no hardware
execution result is inferred from link success.

Source the CANN `set_env.sh` before building: its host-stub extraction calls
`llvm-objdump` through PATH. This CANN installation also exhibited an incremental
preprocess relink failure (`unknown file type`) after overwriting its own object
with raw text. Final device verification therefore used fresh build trees; this
toolkit limitation is not represented as a successful incremental-build check.

Complete configure/build/host logs and the target-manifest audit are available
in `validation_refactor/`. The repository-wide `format.sh ci` check remains
unavailable because pre-commit is not installed; targeted clang-format checks
were completed.
