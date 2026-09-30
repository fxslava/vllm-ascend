---
name: ascend-c-pipeline-sync
description: "Hand-write cross-pipe synchronization on Ascend C under --cce-auto-sync=off: which SetFlag/WaitFlag pair each hazard needs, how to allocate event IDs with FetchEventID instead of hardcoding them, how to prime and drain a flag ring around a software-pipelined loop, TQue vs TBuf and where double buffering actually comes from, and the AIC/AIV split with SyncAll. Use when writing or reviewing any Ascend C kernel that moves data between GM, UB and the compute units."
---

# Memory hierarchy and pipeline synchronization

## Overview

`--cce-auto-sync=off` means the compiler inserts **no** pipe synchronization.
Every cross-pipe dependency is the author's to write, and an omission does not
fail to compile, does not fail on the CPU interpreter, and usually does not
fail the first time on hardware either. This skill is the discipline that makes
omissions visible before silicon.

Check which mode an op is in before reading its kernel — it is in the op's
`op_host/CMakeLists.txt`:

```cmake
add_ops_compile_options(OP_NAME MyOp OPTIONS --cce-auto-sync=off ...)
```

Most ops in `csrc/moe` and `csrc/gmm` are `off`. A few are `on`. The reviewing
standard is completely different between the two.

## The hazard table

Name the *consumer* and the *producer* pipe, in that order, and the flag follows.

| Producer → consumer | Flag | Typical site |
| --- | --- | --- |
| `DataCopy` GM→UB, then read in vector | `MTE2_V` | load then compute |
| `DataCopy` GM→UB, then read with `GetValue` | `MTE2_S` | load a tiling scalar or group list |
| Scalar write to UB, then vector read | `S_V` | build a constant, then use it |
| Vector write, then scalar read | `V_S` | `ReduceMax`, then read element 0 |
| Vector write, then `DataCopy` UB→GM | `V_MTE3` | compute then store |
| Scalar write to UB, then `DataCopy` UB→GM | `S_MTE3` | fill an output row, then store |
| `DataCopy` UB→GM issued, then overwrite that UB | `MTE3_V` / `MTE3_S` | **write-after-read**, buffer reuse |
| Read of a UB buffer done, then `DataCopy` refills it | `V_MTE2` / `S_MTE2` | **write-after-read**, buffer reuse |

The bottom two rows are the ones that get missed. A read-after-write hazard
produces obviously wrong data and gets caught; a **write-after-read** hazard on
a reused staging buffer produces *mostly* right data and gets caught in
production. Any buffer written by MTE2 more than once in a kernel needs an
`S_MTE2` or `V_MTE2` before every refill after the first.

## Hard rules

1. **Every `SetFlag` has a matching `WaitFlag` on every path, including the
   early-return paths.** A flag set and not waited leaks a hardware event; the
   next kernel using that id deadlocks or, worse, does not.
2. **Allocate event ids, do not hardcode them.**
   ```cpp
   int32_t evt = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
   SetFlag<HardEvent::V_S>(evt);
   WaitFlag<HardEvent::V_S>(evt);
   ```
   (`grouped_matmul_swiglu_quant.h:503-512`.) Hardcoded ids are only acceptable
   in a leaf helper that is provably not nested inside another user of the same
   pair. Each `HardEvent` pair has its own id namespace, so `MTE2_S` id 0 and
   `S_MTE3` id 0 do not collide — but two different `MTE2_S` users both using
   id 0 do.
3. **A software-pipelined loop primes its flags before the loop and drains them
   after.** The prologue `SetFlag`s and the epilogue `WaitFlag`s are what makes
   rule 1 hold across the loop boundary.
4. **Do not barrier a pipe against itself.** `PIPE_V` is in-order, so a
   `PipeBarrier<PIPE_V>()` between consecutive vector instructions — including
   a dependent pair reading and writing one buffer — buys no ordering and
   forces a pipeline flush. It is also redundant immediately before a
   `SetFlag<HardEvent::V_*>`, which already orders the vector work ahead of the
   consumer. Barriers belong on heterogeneous boundaries only.
   `PipeBarrier<PIPE_ALL>()` is the exception that is sometimes right: it
   serialises everything, so it belongs around `pipe->Reset()` and at phase
   boundaries, never inside a loop.

   Note the in-repo tension before applying this to an `--cce-auto-sync=off`
   kernel: `chunk_kda_fwd.cpp:399-406` ships with auto-sync off and *does*
   barrier between `Mins` and `Maxs` on one tensor. Either the V pipe
   interlocks same-buffer RAW in hardware and those barriers are cargo, or they
   are load-bearing there. The settling experiment is in the
   `ascend-c-fused-swiglu` skill under "Where barriers belong".
5. **Number your flags in comments.** The reference does this and it is the only
   reason the pairing is reviewable.

## Idiom: primed and drained flag ring

From `grouped_matmul_swiglu_quant_a8w4_msd_pre.h:157-208`, trimmed. Note the
prologue, the rotation inside the loop, and the epilogue that closes every flag
the prologue opened.

```cpp
SetFlag<HardEvent::V_MTE2>(EVENT_ID0);   // 0
SetFlag<HardEvent::MTE3_V>(EVENT_ID0);   // 1
SetFlag<HardEvent::MTE3_V>(EVENT_ID1);   // 2

for (uint32_t i = 0; i < taskNum; i++) {
    WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);   // 0   UB free for a refill
    DataCopy(xTensor, xGM[addr], vK);
    SetFlag<HardEvent::MTE2_V>(EVENT_ID0);    // 3
    WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);   // 3   load landed

    Cast(...);        // no barrier between these two: PIPE_V is in-order
    Muls(...);

    WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);   // 2   previous store drained
    Cast(highI4, ...);
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);    // 4
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);   // 4
    DataCopy(yGm[addr], highI4.ReinterpretCast<int8_t>(), vK / 2);

    SetFlag<HardEvent::MTE3_V>(EVENT_ID1);    // 2   re-arm
    /* ... low-nibble half, symmetric ... */
    SetFlag<HardEvent::V_MTE2>(EVENT_ID0);    // 0   re-arm
}

WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);   // 0
WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);   // 1
WaitFlag<HardEvent::MTE3_V>(EVENT_ID1);   // 2
```

## TQue, TBuf, and where double buffering comes from

| | `TQue<QuePosition::X, N>` | `TBuf<TPosition::VECCALC>` |
| --- | --- | --- |
| Lifecycle | `AllocTensor` → `EnQue` → `DeQue` → `FreeTensor` | `Get<T>()` / `GetWithOffset<T>()` |
| Inserts sync | yes, on `EnQue`/`DeQue` | **no** |
| Use for | anything crossing a pipe boundary | scratch used by one pipe |

**A kernel built entirely from `TBuf` has no implicit synchronization anywhere.**
That is a legitimate design — the reference's pre stage does it and hand-writes
every flag — but it must be a deliberate choice, and it must be stated in the
file header, because a reader who assumes `TQue` semantics will misread every
buffer reuse in the file.

**The buffer count comes from `InitBuffer`, not the template parameter.** This
declares a queue and then gives it two buffers:

```cpp
TQue<QuePosition::VECIN, 1> mmOutQueue;                      // template arg is 1
pipe->InitBuffer(mmOutQueue, DOUBLE_BUFFER, tileBytes);      // ...but two buffers
```

Double buffering only hides latency if the two buffers are used in alternating
iterations. Allocating `DOUBLE_BUFFER` and then `AllocTensor`ing once before the
loop, as `VecProcess` does, gives capacity without pipelining — which is correct
there because the pipelining is at the phase level instead.

L1/L0 staging depth for the Cube is set in tiling, not in the kernel:
`DEPTH_A1`, `DEPTH_B1` (L1 buffer depth), `STEP_Ka`, `STEP_Kb` (how many K
steps are resident), `BASIC_M/N/K` (the L0 tile)
(`grouped_matmul_swiglu_quant_utils.h:53-61`).

## AIC / AIV split

One kernel source serves both cores. Guard with `ASCEND_IS_AIC` /
`ASCEND_IS_AIV` and rendezvous with `SyncAll`:

```cpp
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);   // 1 Cube : 2 Vector

if ASCEND_IS_AIC { /* GEMM, writes int32 to a GM workspace */ }
SyncAll<false>();
if ASCEND_IS_AIV { /* reads that workspace, dequant + activation + quant */ }
```

- The Cube→Vector handoff is through a **GM workspace**, not UB
  (`grouped_matmul_swiglu_quant.h:183`, `mm.IterateAll(mmOutGM[...])`). Sizing
  that workspace is a tiling responsibility.
- `SyncAll<false>()` is the asynchronous form; `SyncAll<true>()` blocks. The
  three-stage software pipeline in `grouped_matmul_swiglu_pipeline.h:89-135`
  runs `pre(n+1) ‖ mid(n) ‖ post(n-1)` between `SyncAll` barriers.
- `pipe->Reset()` between stages re-partitions UB so the stages do not have to
  co-exist in the budget.

## How to verify

Stated plainly, because this is where the effort usually stops too early:

- **The CPU interpreter (tikicpulib) cannot validate any of this.** It runs the
  kernel serially in one process and reports `[TmSim]: Run in serial mode.`; a
  kernel with every flag deleted passes it byte-identically. A pass there is a
  statement about arithmetic and nothing else.
- **The camodel models the pipes** and is the cheapest tier that can expose a
  missing flag — but a scalar-heavy kernel can take tens of minutes per launch
  on it, so budget for that or shrink the geometry.
- **On silicon, the signature of a missing flag is non-determinism**: run the
  same launch with identical inputs N times and compare outputs bit-for-bit.
  A sync defect shows up as a diff between runs, which no single-run
  correctness check will ever catch. This is cheap and it is the test that
  actually earns its keep — see the stress case in
  `csrc/tests/` for the dsv4 expert.
- **Review by flag pairing**: for each `SetFlag` in the file, find its
  `WaitFlag`, on every path. If that takes more than a minute, the numbering
  comments from rule 5 are missing.
