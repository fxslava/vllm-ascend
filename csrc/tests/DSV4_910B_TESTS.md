# DeepSeek-V4 expert verification on Ascend 910B

The dedicated executable uses the shipping kernel and the shared C++ oracle.
It accepts physical Ascend910B1/B2/B3/B4 on device 0, rejects CAModel, and fails
when no usable device is present. Missing hardware is never reported as a pass.

## Build and run

```bash
source /usr/local/Ascend/cann/set_env.sh
cmake -S csrc/tests -B build_910b_test \
  -DASCEND_HOME_PATH=/usr/local/Ascend/cann \
  -DSOC_VERSION=Ascend910B1 -DRUN_MODE=npu \
  -DVLLM_ASCEND_TESTS_WERROR=ON
cmake --build build_910b_test --target test_device_910b_dsv4_moe_expert -j4
ASCEND_TEST_DEVICE_ID=0 build_910b_test/device_910b/test_device_910b_dsv4_moe_expert
```

Use the installed chip's B1/B2/B3/B4 variant for `SOC_VERSION`. The target is
registered with CTest as one physical-device test with a 3600-second timeout.
Kernel compilation uses `--cce-auto-sync=off` and `-Werror`. TurboQuant arch35
probes and the 310P device suite are excluded from the 910B configuration.

## Architecture review

The kernel uses one AIV and no Cube resources. At hidden=7168/inter=2048 its
explicit TBuf allocations are:

| Allocation | Bytes |
| --- | ---: |
| BF16 input | 14336 |
| FP32 input / clamp scratch / down accumulator | 28672 |
| FP32 gate, up, activated | 24576 |
| Three BF16 output rows | 12288 |
| Packed weight staging | 2048 |
| Padded scale staging | 256 |
| Three decode stages and products | 65536 |
| Even and odd input columns | 2048 |
| Two aligned reduction-slot buffers | 16384 |
| Two merge buffers and decoded scales | 3072 |
| Predicate scratch | 4096 |
| BF16 chunk output | 256 |
| **Total** | **173568** |

This is 169.5 KiB, leaving 23040 bytes within a conservative 192 KiB UB budget.
Every allocation is a multiple of 32 bytes. The 8x512 working set remains
unchanged. Weight bursts and input/output rows use 32-byte boundaries; scales
occupy independent padded 32-byte rows through DataCopyPad. Reduction slots are
32 bytes apart. Aliases are retained only across non-overlapping phases:
input/clamp/down, code bits/products, and scale-decode/merge scratch.
FP32 SwiGLU requires no element scratch in the installed CANN implementation;
internal stack/temporary resources still require hardware validation.

CANN 9.2.0-beta.2 arch220 headers and compilation exposed differences from
arch35; the shipping kernel now selects these paths with `__CCE_AICORE__ == 220`:

- DeInterleave is replaced by vector Gather with byte offsets for even/odd
  FP32 lanes, including every reduction-tree level.
- Byte widening uses uint8 to half to float to int32. All values in [0,255]
  are exact through these conversions; unsupported unsigned integer casts
  are excluded.
- Select operates on FP32 views of bit-pattern buffers, because arch220 Select
  supports half/float rather than INT32 data operands.
- INT32 equality predicates remain supported. The scale-special less-than
  predicate converts [0,255] exactly to FP32 before comparison.
- And/Or use UINT16 views with twice the INT32 element count. The installed
  arch220 calcount implementation reinterprets INT32 as 16-bit lanes without
  doubling the mask count, which otherwise leaves half the buffer untouched.

ReduceRepeat SUM uses FP32; Gather, integer shifts and bitwise operations compile
for arch220. BF16 widening and CAST_ROUND narrowing remain in place. The installed
SwiGLU implementation applies swish to its second operand, preserving the
symmetric gate clamp at +/-10 and FP32 gate/up inputs. FP32 reduction is
reassociated relative to the ascending oracle; it is not bit-exact accumulation
in real arithmetic. The BF16 ULP tests determine acceptable numerical error.

The MTE2_V/V_MTE2 staging ring and V_MTE3/MTE3_V row ring retain their prime,
re-arm and drain. The final MTE3_V drain precedes UB destruction. No event IDs,
cross-pipe ordering, or tiling dimensions were changed. Compilation cannot
establish bank-conflict performance or runtime synchronization correctness.

## Coverage and validation status

The suite runs hidden sizes 704 (partial column tile), 4096, and 7168, each with
inter=2048. It reports FpDiff (maximum BF16 ULP distance) and RateDiff (fraction
exceeding 2 ULP) for all four buffers, requiring <=2 ULP and <=0.01 respectively.
NaN/Inf results are compared by classification and infinity sign.

Cases cover 20 repeated launches, six distinct streams with private HBM storage
over ten rounds, both gate saturation signs beyond +/-100, isolated exhaustive
packed-byte decoding in both nibble positions, a finite scale sweep including
subnormals and 120..124, and all 256 scale bytes in gate projection. The full
scale sweep checks projection specials separately because overflow reassociation
and NaN clamp behavior are not finite ULP comparisons.

Local validation on Windows/WSL with CANN 9.2.0-beta.2:

- Ascend910B1 kernel and executable compiled with Werror. Link verification
  required CANN development driver stubs because the NPU driver is absent.
- Host DSV4 tests: 20 passed.
- Physical execution is unverified: no `/dev/davinci*` device nodes are present.
  Attempting execution with development stubs failed before tests with:

```text
symbol lookup error: /usr/local/Ascend/cann/lib64/libruntime_common.so:
undefined symbol: _ZN12ErrorManager19ATCReportErrMessageESsRKSt6vectorISsSaISsEES4_
```

Consequently no hardware ULP metrics, determinism results, concurrency results,
FPU-trap status, or bank-conflict measurements are available. B2/B3/B4 runtime
validation also remains outstanding. Do not treat compile success as silicon
qualification. Complete configure/build/host/run logs are saved beside this
document in `validation_910b/`.

`bash format.sh ci` could not run on the CRLF checkout. Running its normalized
contents also stopped because pre-commit is not installed; targeted clang-format
was applied to the new test source and `git diff --check` passed.
