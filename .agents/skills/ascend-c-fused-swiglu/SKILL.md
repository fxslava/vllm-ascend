---
name: ascend-c-fused-swiglu
description: "Compute SwiGLU on Ascend C: the DeepSeek-V4 swiglu_limit = 10.0 architectural clamp, the high-level SwiGLU API instead of a hand-rolled exp/sigmoid, the mandatory fp32 -> bf16 downcast before the down projection, gate/up staging when W1 and W3 are physically distinct matrices, and where pipe barriers belong (and do not). Use when a kernel applies SwiGLU, SiLU or a sigmoid between two GEMMs."
---

# Fused SwiGLU and activation buffer management

## Overview

SwiGLU sits between two GEMMs, so it is almost never the arithmetic
bottleneck — but it is where a kernel most easily acquires a pipeline flush, a
scalar fallback, a dtype mismatch at the next GEMM's input, or an `inf`. This
skill is the shape that avoids all four.

## Hard rules

1. **`swiglu_limit = 10.0` is an architectural invariant of DeepSeek-V4, not a
   tunable and not an overflow workaround.** The gate activation is clamped
   symmetrically to ±10 before the activation. A kernel that omits it does not
   implement DeepSeek-V4, and its outputs will not match the model's reference
   regardless of how clean its numerics otherwise are.
2. **Use the `SwiGLU` high-level API.** Do not hand-roll
   `x * (1 / (1 + exp(-x))) * y`. The API is one vector call over the tile; a
   hand-rolled version is at best three vector passes and at worst a scalar
   loop. There is no scalar `expf` in device code — reaching for one is the
   signal that this rule was skipped.
3. **Downcast fp32 → bf16 before the down projection.** The SwiGLU output is
   fp32; the down GEMM's A operand is bf16. The conversion is a vector `Cast`,
   never an element loop.
4. **Do not put `PipeBarrier<PIPE_V>()` between consecutive vector
   instructions.** `PIPE_V` is in-order; a barrier there buys nothing and
   forces a pipeline flush. Barriers belong only on heterogeneous boundaries —
   see [Where barriers belong](#where-barriers-belong).
5. **Gate and up are two GEMMs, not two halves of one.** See
   [Gate and up are distinct matrices](#gate-and-up-are-distinct-matrices).

## Two API facts that cost a debugging cycle each

Both measured against `swiglu_3510_impl.h` on CANN 9.1, and both contradict the
obvious reading of the production reference.

**1. The swish is applied to `srcTensor1`, not `srcTensor0`.**

```cpp
SwiGLU<T, reuse>(dst, s0, s1, beta, n);   // computes  s0 * swish(s1)
```

So the tensor that goes through the activation is the **second** source, and
the plain multiplicand is the first. Read
`grouped_matmul_swiglu_quant.h:470-478` and you would conclude the opposite: it
clamps `src0` two-sided (the pattern you would use for an exponential's input)
and `src1` one-sided. Getting this backwards does not fail, warn, or produce
anything obviously broken — it produces `gate * swish(up)`, which is the right
order of magnitude and completely wrong. The symptom is a large relative error
that survives every alignment and sync check.

**2. `dst`, `src0` and `src1` must all report the same tensor size.**
Not "at least as large as `calCount`" — equal. A scratch view taken from a
buffer sized for `max(hidden, inter)` aborts with
`Input params.GetSize must be equal with each other!` even when `calCount` is
well within every allocation. Take the view with an explicit count:

```cpp
LocalTensor<float> clampedGate = scratchBuf.Get<float>(count);   // not Get<float>()
```

## The DeepSeek-V4 clamp

```cpp
constexpr float kSwigluLimit = 10.0f;   // DeepSeek-V4 architectural constant

// Clamp to scratch, not in place: gate_out is the raw projection, and
// clamping it would change what that output means to the caller.
Mins(clampedGate, gateLocal, kSwigluLimit, len);
Maxs(clampedGate, clampedGate, -kSwigluLimit, len);

SwiGLU<float, false>(activatedFp32, upLocal, clampedGate, beta, len);
```

No barriers between those three: all `PIPE_V`, in order, and each reads what
the previous wrote.

Three consequences worth stating plainly:

- **±10 keeps the exponential far inside range.** `exp(10) ≈ 2.2e4`, nowhere
  near the fp32 overflow at ~88.7. The `inf` path that an unclamped kernel
  relies on — `1/(1+inf)` evaluating to exactly `0` — simply does not arise.
  Any hardware FP-status noise from saturated elements disappears with it.
- **The clamp changes results, and that is the point.** It is part of the model
  definition, so the CPU reference, the goldens and the kernel must all apply
  it or they are implementing different functions. Adopting it on an existing
  kernel is a golden re-baseline, not a drop-in patch.
- **Carry the limit in tiling, not as a literal in the kernel.** The value is
  fixed by the architecture, but plumbing it through the tiling struct is what
  lets a second model with a different limit reuse the kernel, and what makes
  the constant visible to the host-side contract. Hard-coding it buries an
  architectural fact inside device code.

## Gate and up are distinct matrices

In DeepSeek-V4 Flash MoE, `W1` (gate) and `W3` (up) are **physically separate
weight matrices**, in separate HBM slot regions, with independent E8M0 scale
blocks. They are not two column-halves of one concatenated weight, and their
GEMM outputs are not two halves of one Cube result.

This is a genuine difference from the fused-QKV-style layout some other MoE
kernels use, and copying that layout's assumptions is a correctness bug, not an
inefficiency.

So the two projections are computed either as **sequential Cube GEMM
iterations** or on **two streams**, and each writes into its own region of a
contiguous vector staging buffer:

```cpp
// Two GEMMs, two scale sets, two destinations -- but one contiguous staging
// buffer, so SwiGLU reads two adjacent slices rather than gathering.
LocalTensor<float> stage    = stageBuf.Get<float>();          // 2 * len floats
LocalTensor<float> gateLocal = stage;                          // [0, len)
LocalTensor<float> upLocal   = stage[len];                     // [len, 2*len)
```

The distinction that matters: **the GEMMs are separate; the staging is
contiguous.** Contiguous staging is what keeps SwiGLU to one call over adjacent
slices. It does not imply, and must not be taken to imply, that a single GEMM
produced both.

## The fp32 → bf16 downcast

`SwiGLU` writes fp32. The down projection consumes bf16. The conversion is an
explicit vector stage:

```cpp
Cast(downInBf16, activatedFp32, RoundMode::CAST_ROUND, len);
```

- It is a `PIPE_V` instruction, so it needs no barrier after the `SwiGLU` that
  produced its input.
- It **does** need a `V_MTE3` flag before any `DataCopy` that stores it, and a
  `V_MTE2`/`MTE2_V` handshake if the down GEMM reads it back through GM.
- **Check the rounding mode against your golden.** A CPU reference that rounds
  fp32→bf16 round-to-nearest-**even** (the `0x7FFF + lsb` idiom) and a kernel
  that rounds half-away-from-zero will disagree by 1 ULP on exact ties. That is
  inside a 2-ULP gate, so it will not fail the test — it will just sit there
  as an unexplained diff. Decide which mode the contract specifies and assert
  it with a tie-valued test case.

## Where barriers belong

`PIPE_V` is in-order. Consecutive vector instructions need no barrier between
them, including when the second reads what the first wrote. Inserting one
drains the pipe for nothing.

| Boundary | Needed | Mechanism |
| --- | --- | --- |
| Vector → vector | **no** | in-order |
| MTE2 load → vector compute | yes | `SetFlag`/`WaitFlag<HardEvent::MTE2_V>` |
| Vector compute → MTE3 store | yes | `SetFlag`/`WaitFlag<HardEvent::V_MTE3>` |
| Vector → scalar read of the result | yes | `HardEvent::V_S` |
| Around `pipe->Reset()` | yes | `PipeBarrier<PIPE_ALL>()` on both sides |

A `PipeBarrier<PIPE_V>()` immediately before a `SetFlag<HardEvent::V_S>` is
also redundant: the flag already orders the vector work ahead of the scalar
read.

So the whole activation block carries no intra-`PIPE_V` barrier at all:

```cpp
SetFlag<HardEvent::MTE2_V>(evtLoad);
WaitFlag<HardEvent::MTE2_V>(evtLoad);          // gate/up tiles have landed

Mins(gateLocal, gateLocal, kSwigluLimit, len);
Maxs(gateLocal, gateLocal, -kSwigluLimit, len);
SwiGLU<float, false>(activatedFp32, gateLocal, upLocal, beta, len);
Cast(downInBf16, activatedFp32, RoundMode::CAST_ROUND, len);

SetFlag<HardEvent::V_MTE3>(evtStore);
WaitFlag<HardEvent::V_MTE3>(evtStore);          // now MTE3 may read it
DataCopy(downInGm, downInBf16, len);
```

**A caveat to settle before applying this to an `--cce-auto-sync=off` kernel.**
The repository's own auto-sync=off kernels do place `PipeBarrier<PIPE_V>()`
between dependent vector instructions — `chunk_kda_fwd.cpp:399-406`
(`ClampExpInput`) is `Mins` / barrier / `Maxs` / barrier on one tensor, which
is exactly the pattern this skill removes. The reference SwiGLU in
`grouped_matmul_swiglu_quant.h:470-477` does the same, but that op compiles
with `--cce-auto-sync=on`, where the barriers are provably redundant because
the compiler inserts what is needed regardless.

That leaves one open question: whether the V pipe interlocks a same-buffer RAW
hazard in hardware, or whether those auto-sync=off kernels are load-bearing.
The experiment that settles it is cheap — remove the barriers from one
auto-sync=off kernel, run it on the camodel N times against a fixed input, and
compare the runs bit-for-bit. Non-determinism means they were load-bearing.
Until that runs, treat the removal as verified for auto-sync=on and as pending
verification for auto-sync=off.

## Buffer management

- **Size one scratch buffer for the largest consumer and sub-allocate it** with
  `TBuf::GetWithOffset<T>(size, byteOffset)`, rather than one buffer per stage.
  State offsets in bytes; that is what the 32-byte rule is about.
- **When a buffer is aliased across phases, size it for the `max` of the
  phases.** A buffer sized for `hidden` floats and later reused for `inter`
  floats overflows the moment `inter > hidden` — and a geometry check that
  never compares the two will not catch it.
- **`pipe->Reset()` re-partitions the whole UB budget**, so a multi-phase
  kernel need not fit every phase simultaneously. It invalidates every
  outstanding tensor, so it goes between phases, never inside one, with
  `PipeBarrier<PIPE_ALL>()` on both sides.

## Boundary behaviour worth testing

| Input | Expected with `swiglu_limit = 10` | Why it is a test |
| --- | --- | --- |
| `gate` far negative | clamped to −10; activation ≈ `-4.5e-4 * up`, **not** 0 | The clamp's whole observable effect |
| `gate` far positive | clamped to +10; activation ≈ `10 * up` | The other saturation end |
| `gate == 0` | activation `0` exactly, sign of zero preserved | `silu(0) == 0` |
| `up == 0` | activation `0` | Zero propagation through the product |
| `NaN` in either half | `NaN` out, not a silent finite value | `Mins`/`Maxs` tie-breaking on `NaN` is implementation-defined; if the contract is that `NaN` propagates it needs a test |
| exact fp32→bf16 tie | matches the reference's rounding mode | Catches a `CAST_ROUND` vs round-to-nearest-even mismatch |

The first row is the one that changes when the clamp is adopted: an unclamped
kernel returns exactly `0` there, a clamped one does not. Every golden
containing a saturated element moves.
