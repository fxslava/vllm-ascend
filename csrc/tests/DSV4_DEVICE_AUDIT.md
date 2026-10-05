# DSV4 routed expert: static audit and compile verification

Date: 2026-10-05. Target: Ascend950PR_9599, CANN 9.1.0 and
9.2.0-beta.2. Verification is **compile-only**, following the user's updated
instruction. No correctness test, simulator launch, hardware launch, benchmark,
or profiler was executed. Numerical parity, trap freedom, concurrent execution,
latency, and bandwidth remain unmeasured.

## Changes and numerical contract

- Production correctness suite: `device_950pr/test_device_950pr_dsv4_moe_expert.cpp`.
- Benchmark: `device_950pr/bench_device_950pr_dsv4_moe_expert.cpp`.
- Shared deterministic synthetic generator, dedicated streams, HBM allocations,
  pinned asynchronous D2H copies, and resident launch resources:
  `common/dsv4_device_case.hpp`.
- Same production correctness source is compiled for the simulator as
  `test_sim_950pr_dsv4_moe_expert_production`.
- The existing reduced simulator suite remains available.
- The shipping kernel now streams eight rows by at most 512 columns, accepting
  hidden dimensions through 7168 and intermediate dimensions through 2048.
  Both dimensions must be positive multiples of 64. A 704-column test covers
  the final partial column tile; production cases use 4096/2048 and 7168/2048.
- The original staging event ring lacked its per-iteration re-arm. This is
  repaired, and the final output DMA is explicitly drained before UB release.
- Gate/up accumulate in FP32. The gate is clamped symmetrically to +/-10 before
  SwiGLU. Activation is rounded to BF16 with `CAST_ROUND`, then widened exactly
  for the down projection. The C++ oracle and Python golden generator use the
  same BF16 boundary. Existing golden files need regeneration for this contract.

For healthy inputs the device suite asserts a maximum of 2 BF16 ULP and a
fraction beyond that tolerance of at most 1e-2, separately for all four outputs.
The maximum-ULP requirement is deliberately strict; the rate assertion does not
relax it. There is no claim that the newly reassociated reduction passes until
these tests execute.

Weights cover every packed byte, including all 16 E2M1 codes in both nibble
positions. The finite random cases use E8M0 exponents 120 through 124. A separate
production projection test sweeps every exponent byte 0 through 254 and byte
255. E8M0 has **no infinity code**: 255 is NaN, 0 is 2^-127, and 254 is 2^127.
The latter produces arithmetic overflow with a positive E2M1 weight. Specials
are checked by classification/sign in the gate projection, separately from
finite ULP parity; downstream NaN clamp behavior is not asserted as finite
parity. Extreme-gate cases exercise both signs past +/-100 and check finite
outputs against the clamped oracle.

Six experts receive different seeds, separate HBM resources, tiling buffers,
and streams. All six launches are submitted before any readback synchronization.
Repeated identical launches compare all four buffers bit-for-bit.

## Static analyzer

Analyzer: `C:\ascend-kernel-analyzer`, invoked from Ubuntu-22.04 WSL using its
installed Python entry point. The missing `pcpp` dependency was installed.
Solver: Z3. Chip profile: **provisional `ascend351x`**, because this analyzer has
no authoritative 950PR SKU profile. Conclusions about bank geometry and SRAM
capacity use that profile, not a measured SKU configuration.

The initial source-only audit used role inference, which supplied reduced
64/64 dimensions. It reported 1 fatal and 38 warnings. A final audit explicitly
binds 7168/2048 through `reference/dsv4_production_tiling.json`; no reduced
geometry inference is used for that report.

```bash
ascend-analyze csrc/moe/dsv4_moe_expert/op_kernel/dsv4_moe_expert.cpp \
  --chip ascend351x --solver z3 --all-functions \
  --tiling-data csrc/tests/reference/dsv4_production_tiling.json \
  --no-color --ascii --max-findings 0 \
  --json build/dsv4-device-audit/analyzer-production.json \
  -o build/dsv4-device-audit/analyzer-production.txt
```

The final production-bound report contains **0 fatal, 28 warnings, 4 info**.
No diagnostic was suppressed. Counts include analysis of helpers and the
inlined entry point; they are not counts of unique physical defects.

| Code | Initial count | Final production count | Assessment |
| --- | ---: | ---: | --- |
| AKA2006 | 1 fatal | 0 | Real missing `V_MTE2` re-arm; fixed. |
| AKA1009 | 2 warning | 2 warning | Analyzer cannot resolve the domain of the `legAcc` slice and a reinterpreted decode parameter. Both originate in VECCALC TBuf storage; manual review is still necessary. |
| AKA2010 | 4 warning | 4 warning | Analyzer classifies stores to indexed GM tensor arrays as MTE2 loads. These are UB-to-GM MTE3 stores, preceded by V_MTE3 handshakes. Do not add the suggested V_MTE2 handshake to these stores. |
| AKA3002 | 32 warning | 6 warning | Remaining unresolved sizes for row slices, merge view, and output subviews. Capacity, alignment, and aliasing are not fully proved by the tool. |
| AKA3006 | 0 | 14 warning | Same-bank source bases in dequantization, input multiplication, merge adds, and scale decoding. Unresolved initial layouts prevented this checker from running there. These are residual performance risks. |
| AKA4001 | 0 | 2 warning | Modeled exposed synchronization stalls, particularly V_MTE3 and V_MTE2. This is an analytical estimate, not measured device timing. |
| AKA3004 | 0 | 4 info | Unclaimed ranges in per-function tensor maps. Functions use subsets of the allocated buffers; these messages do not mean the allocator actually leaves those ranges unallocated. |

Complete diagnostics, locations, explanations, dependency graphs, and model
output are retained in:

- [Initial JSON](../../build/dsv4-device-audit/analyzer-before.json)
- [Initial text](../../build/dsv4-device-audit/analyzer-before.txt)
- [Final production JSON](../../build/dsv4-device-audit/analyzer-production.json)
- [Final production text](../../build/dsv4-device-audit/analyzer-production.txt)

### Pipe and marked-graph review

The observed graph uses MTE2, Vector, and MTE3. The source is AIV-only and does
not issue Cube contractions. The AIC early return occurs before pipe creation
or event issuance. There are no other early exits in the compute path.

| Route and ID | Pairing invariant |
| --- | --- |
| MTE2_V / 3 | One input-load set/wait. |
| MTE2_V / 0 | One set/wait per weight/scale column tile through `WaitLoad`. |
| V_MTE2 / 0 | One prime, one wait and re-arm per column tile, one final drain. Protects staging-buffer WAR reuse. |
| MTE3_V / 1 | One prime, one wait and re-arm per output row tile, one final drain. Protects row-staging WAR reuse. |
| V_MTE3 / 1 | One set/wait before each padded output-row store. |
| V_MTE3 / 2 | One set/wait before the final three output-row stores. |
| MTE3_V / 2 | One final set/wait drains those output stores. |

IDs 0 through 3 stay outside the analyzer's reserved IDs 6 and 7 and are distinct
within each event namespace used by these helpers. No direct MTE2_S, S_MTE2,
S_MTE3, S_V, or V_S occurs in this kernel. The marked-graph analysis found no
token-free circular wait after the ring repair. These static checks do not
prove freedom from hardware deadlocks under load.

Inspection of CANN's `dav_3510/kernel_operator_vec_reduce_impl.h` showed that
the previous level-2 `ReduceSum` performs a V_S handshake and scalar load of
`dstLocal[0]` for every partial reduction. The new path uses
`ReduceRepeat<ReduceType::SUM>` and a vector gather, avoiding that wrapper's
scalar read. Scalar loop/control instructions remain. The compiler's expanded
device flags contain both `--cce-auto-sync=off` and `-Werror`.

### UB layout, alignment, and remaining bottlenecks

At 7168/2048 the explicit InitBuffer allocations total **173568 bytes
(169.5 KiB)**. This excludes compiler reservation and any high-level API scratch.
The provisional 256 KiB profile leaves 86576 bytes before those reservations;
the explicit footprint is also below its 216 KiB SIMT tensor limit. This kernel
uses the AIV vector path, not a SIMT launch.

| Allocation group | Bytes |
| --- | ---: |
| Input BF16 bits and reusable FP32 input/down accumulator | 43008 |
| Gate/up/activation FP32 buffers | 24576 |
| Three final BF16 intermediate rows | 12288 |
| Packed weight and padded scale staging | 2304 |
| Three decode stages and product buffer | 65536 |
| Even/odd input tile buffers | 2048 |
| Two sparse aligned reduction buffers | 16384 |
| Two merge buffers and decoded scales | 3072 |
| Decode predicate and per-row BF16 staging | 4352 |

Every allocation base/extent is a multiple of 32 bytes. Packed DMA rows are
burst-aligned. Scale DMA rows use DataCopyPad into separate 32-byte slots;
unaligned GM scale offsets and 16-byte output tails use padded-copy APIs rather
than ordinary DataCopy. Reduction destinations now also use 32-byte slots,
compacted by a byte-offset gather. Maximum gather offset is 8160 bytes inside
an 8192-byte reduction buffer. The final 512-column tile may be shorter, but its
width remains a multiple of 64.

The production-bound analyzer resolves enough layout to find identical-bank
bases, for example products at 0x20100 versus even/odd inputs at
0x24100/0x24500, and merge buffers at 0x28900/0x28d00. Full-tile row strides
are 1024 bytes, preserving the same bank phase across rows. No distinct
512-byte stride diagnostic was emitted; absence of such a diagnostic is not
proof that every access pattern is conflict-free. A future measured tuning pass
should evaluate 32-byte bank-skew padding against the target SKU's actual bank
mapping.

Other bottlenecks remain: single-AIV occupancy, no Cube work, serial dependency
between projection legs, synchronous tile handoffs, repeated vector passes for
FP4 decoding, per-block reductions, and redundant gate/up row stores before
final output copies. The analytical model does not fully expand SDK internals
or model cache behavior. It cannot supply TFLOPS utilization or bandwidth
saturation percentages. Do not quote its modeled cycles as hardware latency.

## Compile verification

| Configuration | Targets | Result |
| --- | --- | --- |
| CANN 9.1.0, sim, Ascend950PR_9599 | Reduced correctness, production correctness, simulator-linked benchmark | Compile and link passed with -Werror. |
| CANN 9.2.0-beta.2, sim, Ascend950PR_9599 | Same three targets | Compile and link passed with -Werror. Toolkit emitted a missing driver-library linker warning; no binary was run. |
| CANN 9.2.0-beta.2, npu, Ascend950PR_9599 | Device correctness and benchmark | Compile and link passed with -Werror using toolkit HAL development stubs. This is not driver/runtime validation. |
| Host C++ reference regression source | test_host_dsv4_moe_expert.cpp | -Wall -Wextra -Werror syntax compilation passed; not executed. |

CANN 9.1 was built in the existing `rfc019` container. CANN 9.2 was built in
Ubuntu-22.04 WSL. Identical kernel/test sources were copied to Linux-local
temporary trees. Building from the Windows mount produced clock-skew/repeated
preprocess-link failures in CANN; clean Linux-local build directories avoided
that toolchain problem. The device link additionally required the toolkit's
HAL development library and error-manager dependencies. No project RPATH was
changed to add simulator libraries to a device executable.

Complete configure/build logs are under `build/dsv4-device-audit/`, named
`configure-9.1-sim.log`, `build-9.1-sim.log`, `configure-9.2-sim.log`,
`build-9.2-sim.log`, `configure-9.2-npu.log`, and `build-9.2-npu.log`.
The final BF16 GroupedMatmulV5 benchmark compile/link checks are recorded in
`benchmark-final-9.1-sim.log`, `benchmark-final-9.2-sim.log`, and
`benchmark-final-9.2-npu.log`. Host-only configuration also passed.

For a configured CANN environment, compile the simulator suites without running
them as follows. Use a fresh build directory if CANN's preprocess linker fails
on an already-merged object.

```bash
cmake -S csrc/tests -B /tmp/dsv4-sim-build \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann \
  -DSOC_VERSION=Ascend950PR_9599 -DRUN_MODE=sim \
  -DVLLM_ASCEND_TESTS_WERROR=ON
cmake --build /tmp/dsv4-sim-build --target dsv4_sim_targets -j4
```

Device builds use `-DRUN_MODE=npu` and target `dsv4_device_targets`.
Host-only mode still excludes both device suites. The production simulator
target and simulator-linked benchmark are not registered for default ctest
execution. The benchmark refuses camodel timing at runtime.

## Benchmark definition and measured results

The baseline uses official CANN **aclnnGroupedMatmulV5**, Clamp, Cat,
SwiGlu, and Cast APIs. All three GEMMs use pre-dequantized BF16 inputs/weights
with FP32 output tensors, preserving the unrounded gate/up values required by
this kernel's activation contract. The activation is explicitly cast to BF16
before the down GEMM. There is no precision-changing fallback if the runtime
rejects a BF16-input/FP32-output tuple: planning or parity failure aborts timing.
The tuple still requires validation on the actual device runtime.

The workspace signature is taken directly from the official CANN 9.2 operator
header when available. The installed 9.1 bundle lacks individual operator
headers, so it uses the matching explicit signature; both V5 symbols were
verified in that bundle's libopapi.so. This is ABI wiring and availability
inspection, not an operator execution test. This baseline times dense resident
BF16 GEMMs rather than device-side packed-FP4 dequantization.

Weights are dequantized and uploaded before timing. Each sample receives fresh
one-shot ACLNN executors planned outside the event interval, with a reused
workspace sized to the largest operation. The stream orders workspace reuse.
Both paths perform 50 warmup iterations followed by 200 event-timed samples;
the reported latency is their median. Correctness gates run before timing and
again after the timed sequence; custom output bits must remain identical.

The implementation prints latency, baseline/custom speedup, effective useful
TFLOPS (6*hidden*inter operations), logical minimum GB/s, and owned HBM including
planned workspace. Event intervals exclude uploads/readbacks and workspace
planning, but can include device idle gaps between host submissions. Owned HBM
does not include opaque CANN caches or establish measured process peak memory.
Measured hardware counters and SKU peak specifications are required to report
compute utilization or memory saturation.

| Implementation | Latency (us) | Speedup | Effective TFLOPS | HBM GB/s | Peak HBM |
| --- | --- | --- | --- | --- | --- |
| Custom dsv4_moe_expert | Not run | Not measured | Not measured | Not measured | Not measured |
| Standard dense BF16 ACLNN V5 baseline | Not run | Not measured | Not measured | Not measured | Not measured |

No FpDiff, RateDiff, hardware trap, six-stream isolation, or performance pass is
claimed. Full-repository test execution and device profiling were intentionally
omitted under the compile-only instruction. The residual analyzer warnings and
the absence of runtime validation prevent a production-readiness claim.
