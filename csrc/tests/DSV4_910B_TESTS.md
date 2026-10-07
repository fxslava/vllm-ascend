# DeepSeek-V4 expert verification on Ascend 910B

The dedicated executable uses the shipping kernel and the shared C++ oracle.
It accepts physical Ascend910B1/B2/B3/B4 on device 0, rejects CAModel, and fails
when no usable device is present. Missing hardware is never reported as a pass.

## Operator registration (2026-10-07)

Until this change the operator was reachable on the Ascend950 only: both
`op_host/CMakeLists.txt` and `dsv4_moe_expert_def.cpp` named `ascend950` alone,
so an `ascend910b` build skipped the op entirely and no amount of kernel
correctness made it dispatchable. It is now registered and built for both:

- `op_host/CMakeLists.txt` compiles the op when either compute unit is
  requested, and `config/ascend910b/dsv4_moe_expert_binary.json` mirrors the
  ascend950 binary config.
- `dsv4_moe_expert_def.cpp` adds one `OpAICoreConfig` for `ascend910b`
  (Ascend910B1..B4) beside the `ascend950` one. One kernel source serves both.
- `op_kernel` classifies the core generation once, in
  `dsv4_moe_expert_vector_compat.h`: `__CCE_AICORE__` 220 is the dav-c220
  vector core and 310 the dav-c310 regbase core, and `DSV4_ARCH_C220` carries
  that choice to every arch-dependent branch. A third generation is now a
  compile error instead of silently inheriting the regbase path, which is what
  the previous bare `#else` did.
- `CANN_VERSION_MAJOR` still selects WholeReduceSum/CompareScalar against
  ReduceRepeat/Compares, but its fallback follows the target rather than
  assuming CANN 8 everywhere. Nothing ships a CANN 8 toolkit for the regbase
  core, so an op-package build that omits the macro no longer compiles the
  CANN 8 reduction path for the Ascend950.
- The two parts do not have the same UB per core (192 KiB on the 910B, more on
  the 950). `Tiling4Dsv4MoeExpert` now computes the kernel's own InitBuffers
  total and checks it against `GetCoreMemSize(UB)`, falling back to 192 KiB
  when the platform cannot be queried. At hidden=7168/inter=2048 that total is
  173568 bytes, so production clears the 910B with 23040 bytes to spare; the
  dimension limits, which were picked against the 950, are no longer the only
  guard. `csrc/tests/common/dsv4_test_oracle.hpp` computes the same total for
  the host tier and `test_host_dsv4_moe_expert.cpp` asserts it.

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
- Byte widening casts uint8 -> half (`vconv_u82f16`) and half -> int32
  (`vconv_f162s32r`); 0..255 is exact in half, so the pair is lossless. It
  replaces a Gather over the packed buffer, which was wrong: Gather bounds its
  source by the extent of the tensor it is handed, and a uint8 staging buffer
  reinterpreted as int32 declares a quarter of the bytes it holds, so byte
  offsets at or past that quarter read undefined data. With the 2 KiB weight
  staging that cut in at byte 512 -- under three of the eight rows in a chunk
  at the production column tiling -- and the 256-byte scale staging was past
  its 64-byte limit from the first block.
- E2M1 and E8M0 predicates use INT32 bounds, subtraction, and bitwise masks.
  Boolean values expand to all-zero or all-one words before masking IEEE-754
  fields. The arch220 decoder does not use Select or FP32 predicate views.
- E2M1 mask scratch reuses full-sized decode stages. The packed predicate
  buffer is too small to hold an INT32 predicate for every decoded weight.
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

### CAModel full-size validation (2026-10-07)

The standalone `test_dsv4_expert` binary accepts `hidden inter seed` arguments
(defaults 64 64 704) so the CAModel can run production geometries directly; the
pass gates are FpDiff <= 2, zero mismatches, zero non-finite outputs. Measured
under CAModel (CANN 8.0.0, Ascend910B1 sim):

- hidden=64/inter=64: all four buffers FpDiff=0, exact parity (~3 min).
- hidden=704/inter=2048: all four buffers FpDiff=0, exact parity (~5 h).
- The exact-parity path is a chain-accumulator reduction: each 64-row pass
  decodes one four-column block and accumulates the columns sequentially
  (even 2k then odd 2k+1) in ping-pong fp32 sums, reproducing the host
  oracle's accumulation order bit-for-bit. The down row leaves through one
  bulk cast+copy in CopyOut (the chain leg no longer stores per-chunk).
  Bring-up on the CANN 8.0.0 arch220 calcount backend exposed toolchain
  hazards that shaped the design, all now encoded in the op's structure:
  plain 1D DataCopy GM->UB no-ops at blockLen 1 and 16 while DataCopyPad
  blockLen 2/16 work; Gather mis-executes with unaligned byte offsets and
  self-tramples when its destination aliases its offset tensor; calcount
  ops at counts outside the verified envelope (below 8, or above 256 for
  the decode/ALU ops) silently mis-execute, so every op above runs at a
  count the tree kernel already verified; long-lived UB views must be
  rebuilt per block because interposed gathers trample them.
- 576/4096/7168 geometries: pending CAModel runs (hours to days each).

Per-geometry runs need separate CAMODEL_LOG_PATH directories. The runtime
LD_LIBRARY_PATH must include /opt/arm64-libs:/usr/aarch64-linux-gnu/lib for
the aarch64 bisheng and simulator under QEMU.

The chain-accumulator description in the bullet above does **not** match the
committed kernel, which streams 8 rows by at most 512 columns, reduces each
32-element block with WholeReduceSum/ReduceRepeat and folds the partials in a
power-of-two tree, and stores the down row per 8-row chunk inside `ProjectLeg`
rather than in bulk from `CopyOut`. Treat that bullet's design claims, and the
parity figures attached to them, as describing something other than
`op_kernel/dsv4_moe_expert.cpp` at this commit.

### Dual-architecture CAModel validation (2026-10-07)

CANN 9.2.0-beta.2 ships x86_64 CAModels for both parts under
`tools/simulator/Ascend910B4` and `tools/simulator/Ascend950PR_9599`, so the
same kernel source can be executed for each core generation natively, without
QEMU and without hardware. The standalone `test_dsv4_expert` harness was built
twice (`-DSOC_VERSION=Ascend910B4` and `-DSOC_VERSION=Ascend950PR_9599`,
`RUN_MODE=sim`, `--cce-auto-sync=off -Werror`, `CANN_VERSION_MAJOR=9`) and run
against the shared C++ oracle. Gates: FpDiff <= 2, zero mismatches, zero
non-finite outputs.

| SoC | hidden | inter | Column tiles as run | Result |
| --- | ---: | ---: | --- | --- |
| Ascend910B4 | 64 | 64 | 1 x 64 | FpDiff=0 on all four buffers |
| Ascend910B4 | 576 | 64 | 512 + 64 | FpDiff=0 on all four buffers |
| Ascend910B4 | 64 | 576 | 512 + 64 on the down leg | FpDiff=0 on all four buffers |
| Ascend910B4 | 1024 | 64 | 512 + 512 | FpDiff=0 on all four buffers |
| Ascend950PR_9599 | 64 | 64 | 1 x 64 | FpDiff<=1 (gate/up), 0 mismatches |

The 950PR's FpDiff=1 on gate/up is fp32 reassociation against the oracle's
ascending walk, not a decode difference: `activated` and `down`, which consume
those values, are bit-exact. Run times on this host are roughly 45 s for
910B4/64x64, 200-620 s for the larger 910B4 geometries, and 6 min for
950PR/64x64 -- the 950PR CAModel is about an order of magnitude slower per unit
of work.

**These runs predate the balanced column tiling below**, so the tile widths in
that column are the greedy ones. The multi-tile rows would now run 320 + 256
and 512 + 512, which regroups the fp32 block partials: re-running them should
stay inside the <= 2 ULP gate but is not expected to reproduce FpDiff=0
bit-for-bit. Nothing about decode, scaling or synchronisation changed.

Two further checks, with the same gates:

- `CANN_VERSION_MAJOR=8` on CANN 9.2 selects WholeReduceSum/CompareScalar
  instead of ReduceRepeat/Compares. It still compiles and still produces
  FpDiff=0 on the 910B4 at 64x64, so the old fallback was not a wrong-answer
  bug -- but it did silently pick the CANN 8 reduction for an Ascend950
  op-package build, which is why the fallback now follows the target.
- The core-generation guard was exercised by preprocessing the arch header for
  `dav-c220`, `dav-c310`, `dav-m200` and `dav-m200 -DASCENDC_CPU_DEBUG=1`:
  c220 selects the c220 branch, c310 and the host-stub pass select regbase,
  m200 is rejected with the `#error`, and m200 under CPU_DEBUG is exempt.

### Balanced column tiling (2026-10-07)

`ProjectLeg` used to claim `MAX_CHUNK_COLS` (512) greedily and leave the
remainder as a final tile, so a 704-column leg ran 512 + 192. That second tile
carried six of sixteen blocks and still paid a whole tile's fixed reduction
cost: the slot-buffer zeroing, the 32-block scale decode, the slot gather and
the power-of-two merge tree are all sized by `pow2Blocks`, never by
`activeBlocks`. `BalancedChunkCols` now spreads the columns over the same
number of tiles, so 704 runs 384 + 320 and 576 runs 320 + 256.

The width has to stay a multiple of `COL_TILE_GRAIN` (2 * FP4_BLOCK = 64),
which is why the even split of 704 is 384 + 320 and not 352 + 352:
`activeCols / FP4_PER_BYTE` is the packed-weight DataCopy's burst length and,
through `col0`, its UB and GM offsets. At 352 that burst is 176 bytes -- 5.5
blocks, so it over-reads 16 bytes past the tile and lands every odd staging row
off a 32-byte boundary.

Two things this does **not** do. It does not reduce total work: the tile count
is deliberately unchanged, the per-column work is unchanged, and the fixed
per-tile cost is still `pow2Blocks`-sized, so the sum over tiles is the same.
What it removes is one tile carrying almost none of the work that cost buys,
and with it the narrowest DMA burst (96 bytes at 704, now 160). And it does not
preserve bit-exactness: regrouping which blocks fold together in the fp32 merge
tree moves results within the <= 2 ULP contract.

The measurable item is the over-padding itself. No tile can exceed
`MAX_CHUNK_COLS / FP4_BLOCK` = 16 blocks, yet `pow2Blocks` is `BYTES_ALIGN` =
32, so every tile decodes 256 scale bytes to use at most 128, zeroes 2 x 2048
reduction slots to use at most 1024, and runs one extra merge-tree level.
`pow2Blocks` is pinned at 32 because it is simultaneously the scale staging's
row stride in bytes, and `DataCopyPad` needs that destination 32-byte aligned.
Decoupling the staging stride from the merge stride would halve the fixed cost,
but it moves the reduction layout and wants CAModel confirmation, so it is not
done here.

The tiling arithmetic is mirrored as `ExpertChunkCols` in
`common/dsv4_test_oracle.hpp` and checked by enumeration over every reduction
dimension the tiling function accepts, in
`host/test_host_dsv4_moe_expert.cpp`: legal grain, the staging and decode
bounds, exact coverage, the minimum tile count, minimality of the width, and
that the last tile is never narrower than the greedy split left it. A 64-column
grain cannot always reach an even split -- `fullCols=4160` needs nine tiles and
`ceil(4160/9) = 463` rounds back up to 512 -- so "never worse than greedy" is
the invariant that holds everywhere, not evenness.

Run CAModel geometries one at a time. Concurrent CAModel instances on this
host lost their entire stdout on three occasions while the simulated work and
the per-core dumps completed normally; the dmesg ring shows unrelated
`libstars.so` teardown segfaults from other CAModel runs on the same machine.
Treat a run with no `geometry`/`FpDiff` output as void and repeat it, with
`stdbuf -o0` and a direct file redirect rather than a pipe.

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
