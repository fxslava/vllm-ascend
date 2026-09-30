# dsv4_moe_expert kernel bring-up

Standalone validation harness for the `dsv4_moe_expert` Ascend C kernel
(`csrc/moe/dsv4_moe_expert/`), at reduced geometry, with no NPU in the loop.

Everything here drives the **shipping kernel source unmodified**. Neither gate
keeps a copy of it: Gate B compiles `op_kernel/dsv4_moe_expert.cpp` as its own
translation unit, Gate C `#include`s it verbatim from a wrapper.

## Layout

| file | role |
| --- | --- |
| `gen_golden.py` | golden generator + comparator; the single source of numeric truth |
| `sim_main.cpp` | Gate B driver — CPU interpreter (tikicpulib) |
| `CMakeLists.txt` | Gate B build |
| `cpu_tiling_shim.h` | supplies `GET_TILING_DATA_WITH_STRUCT` outside the TBE pipeline |
| `gate_c/` | Gate C — device object code on the CANN camodel |
| `run_gate_b.sh` / `run_gate_c.sh` | run one gate end to end, inside the vendor container |

## Running

Both scripts run **inside** the pinned vendor image. From WSL:

```bash
docker run --rm -v $HOME/dsv4-kernel:/work -e SOC_VERSION=ascend950pr_9579 \
  --entrypoint bash quay.io/ascend/vllm-ascend:v0.26.0rc1-a5 \
  /work/tools/dsv4_moe_runtime/kernel_bringup/run_gate_b.sh /work/sim_test 256 128
```

Every file crossing a Windows mount must be CRLF-stripped first
(`find . -type f -exec sed -i 's/\r$//' {} +`) or the shell steps die with
`$'\r': command not found`.

## What each gate can and cannot prove

**Gate B — CPU interpreter.** Compiles the kernel with `ASCENDC_CPU_DEBUG=1`
against `tikicpulib::Ascend950PR_9579` and launches it with
`AscendC::RunKernelFunctionOnCpu`. Seconds to run, no op package, no driver.

It validates **arithmetic only**. The interpreter announces `[TmSim]: Run in
serial mode.` and executes the kernel serially on the host, so a missing
cross-pipe `SetFlag`/`WaitFlag` has nothing to expose — measured: the pre-audit
kernel, which carried no MTE2/MTE3 flags at all, produced byte-identical
output. Do not read a Gate B pass as evidence about synchronisation.

**Gate C — camodel.** Compiles the kernel to real device object code through
`ascendc_library(... RUN_MODE=sim)` and runs it against
`libruntime_camodel.so`. This is the tier that models the pipes, and therefore
the only one that says anything about the kernel's hand-written sync
discipline. It is also where `npusim record` can take a cycle-level trace.

## Traps this harness already works around

* **`GET_TILING_DATA_WITH_STRUCT` is not a header macro.** The TBE op-compile
  pipeline synthesises it per operator (`tbe/tikcpp/get_op_tiling.py`), baking
  the host tiling function's output into the binary as a constexpr byte array.
  It exists only inside `build.sh --pkg`. Any out-of-pipeline build of a kernel
  that uses it needs `cpu_tiling_shim.h`.
* **Two different SoC identifiers.** The toolchain and the camodel directories
  use the part (`Ascend950PR_9599`); `npusim record -s` accepts only the family
  (`Ascend950` / `Ascend950DT`) and rejects the part outright. The simulator
  ships no `Ascend950PR_9579` at all — CANN folds every C310 part onto the 9599
  series (`tikicpulib-config.cmake`, `product_map()`), so 9599 is the
  documented stand-in and Gate C ticks are labelled approximate.
* **One `ascendc_library` source, not two.** The launcher generator emits a stub
  calling `<name>_origin` for every `__global__ __aicore__` signature it sees,
  including a bare *declaration*. Listing the kernel and a separate launcher TU
  that forward-declares it yields two stubs for one name and the merge link
  fails with `undefined symbol: dsv4_moe_expert_origin`.
* **Relative includes in the Gate C wrapper.** `ascendc_library` compiles through
  four nested ExternalProjects; `-I` does not reach every stage, and a forwarded
  `-include` pair arrives at the bisheng host-stub stage as two separate tokens
  (`cannot specify -o when generating multiple output files`).
* **Makefiles, not Ninja.** Under Ninja, CMake writes out-of-tree object paths
  with a `/./` segment that CANN's `extract_host_stub.py` does not normalise;
  the build dies with a bare `KeyError`.
* **Camodel link flags.** `libruntime_camodel.so`'s own `DT_NEEDED` and the
  register symbols `libnpu_drv.so` takes from `libstars.so` resolve at load
  time, so the executable needs `-Wl,-rpath-link,<simdir>` and
  `-Wl,--allow-shlib-undefined`, plus that directory on `LD_LIBRARY_PATH`
  ahead of the toolkit's.

## Pass criteria

`gen_golden.py cmp` reports, per output buffer and overall:

* **FpDiff** — max distance in bf16 ULPs. Pass: `<= 2`.
* **RateDiff** — fraction of elements differing at all. Pass: `<= 1e-2`.
* **MaxRelErr** — max relative error, reported for context.
