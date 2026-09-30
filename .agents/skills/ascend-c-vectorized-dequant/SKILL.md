---
name: ascend-c-vectorized-dequant
description: "Unpack sub-byte quantized weights (FP4/E2M1, INT4, E8M0 block scales) on Ascend C without scalar loops. Covers the vector nibble-split idiom, the arch35 native signed-nibble Cast, letting the Cube consume int4b_t straight from GM, and feeding block scales through SetQuantVector instead of decoding them by hand. Use when a kernel reads packed 4-bit weights or per-block scales."
---

# Vectorized dequantization and staging

## Overview

A packed-4-bit weight path has three jobs: get the nibbles apart, apply the
block scale, and hand the result to a compute unit. Done element-by-element on
the scalar pipe, all three cost roughly an order of magnitude more than the
arithmetic they feed. This skill is the set of idioms that keep them off
`PIPE_S`, extracted from `csrc/gmm/grouped_matmul_swiglu_quant`, which is the
production A8W4 path in this repository.

## Hard rules

1. **No `GetValue`/`SetValue` inside a reduction or element loop.** Those are
   scalar (`PIPE_S`) UB accesses. They are legitimate for reading a handful of
   tiling scalars or a reduction result, and for nothing else. A loop body
   containing one is a defect, not a style preference.
2. **Prefer not unpacking at all.** The Cube reads `int4b_t` directly from GM.
   Unpacking in the vector unit is the fallback for when the operand type or
   layout rules it out — see [Choosing a path](#choosing-a-path).
3. **Block scales go to the Cube, not to a decoder.** `SetQuantVector` applies
   a per-channel scale vector in the Fixpipe. One Cube iteration per scale
   block, accumulating into the same output, beats any hand-rolled dequant.
4. **A `Cast` the hardware does not implement compiles to nothing.** It does not
   error. Compile probes therefore prove nothing about `Cast` support; only a
   numeric check against a CPU reference does. Treat every new
   `Cast(dst, src, mode, n)` type pair as unverified until a test compares its
   output.

## Choosing a path

| Situation | Path |
| --- | --- |
| Weights are integer 4-bit (`int4b_t`) and the Cube is doing the GEMM | Hand the packed GM tensor to the Matmul object as `int4b_t`. No unpack. |
| Weights are E2M1 (FP4) and you are on arch35 | The Cube has an fp4 operand path. Check the operand tuple exists for your dtype combination before committing to it — the supported tuples are narrower than the docs suggest. |
| You need the values in the vector unit (activation-side split, LUT-coded formats) | Vector nibble split, below. |
| The format is a lookup table, not an affine code (E2M1, Lloyd-Max) | Nibble-split to an integer code, then `Gather` through a 16-entry LUT. The code is an index, not a number — do not try to reconstruct it arithmetically. |

**E2M1 is not INT4.** `int4b_t` is a signed integer; E2M1 is a 16-entry
floating-point code whose values are `{0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}`.
The reference's `int4b_t` Matmul path is *not* transplantable to an E2M1 weight
without either the Cube's fp4 operand path or a LUT step. This distinction is
the single most common way an FP4 kernel design goes wrong.

## Idiom: vector nibble split

From `grouped_matmul_swiglu_quant_a8w4_msd_pre.h:144-198`. This splits one
`int8` lane into its two 4-bit halves across a whole vector, with no shifts and
no scalar code.

Build the mask once, outside the loop:

```cpp
Duplicate(maskI16, static_cast<int16_t>(0x0F0F), 128);
```

**Low nibble** — mask it off through an `int16` reinterpretation, so one `And`
covers two lanes:

```cpp
And(lowHalf.ReinterpretCast<int16_t>(), src.ReinterpretCast<int16_t>(), maskI16,
    LEN_128, repeats, {1, 1, 1, 8, 8, 0});
Cast(lowAsHalf, lowHalf.ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, n);
Adds(lowAsHalf, lowAsHalf, static_cast<half>(-8), n);   // unsigned code -> signed
Cast(lowI4, lowAsHalf, RoundMode::CAST_NONE, n);
```

**High nibble** — an arithmetic shift-right-by-4 expressed as a multiply and a
floor, which the vector unit does in two instructions:

```cpp
Cast(highHalf, src, RoundMode::CAST_NONE, n);
Muls(highHalf, highHalf, static_cast<half>(0.0625f), n);  // 1/16
Cast(highI4, highHalf, RoundMode::CAST_FLOOR, n);         // floor == >>4 for the sign
```

Every step above is `PIPE_V` and each reads what the previous wrote, so **none
of them needs a barrier between**: the vector pipe is in-order, and a
`PipeBarrier<PIPE_V>()` there would flush the pipeline for no ordering. The
barriers appear in the reference listing because that op compiles with
`--cce-auto-sync=on`, where they are redundant twice over. Sync is still
required where this block meets another pipe — `MTE2_V` before it, `V_MTE3`
after — and that is all.

**On arch35 there is a shorter route:** the AIV implements a native
signed-nibble `Cast` and a `DeInterleave`, which replace the mask/multiply/floor
pair above. Prefer it where available — but see hard rule 4: verify it
numerically, because an unsupported `Cast` is silently a no-op.

## Idiom: block scales through the Cube

From `grouped_matmul_swiglu_quant_a8w4_msd_mid.h:231-247`. The reduction is
split into one Cube iteration per scale block; the scale vector is handed to the
Matmul object and applied in the Fixpipe; partial products accumulate into the
same output tile.

```cpp
for (uint32_t loopK = 0; loopK < quantGroupNum; loopK++) {
    mm.SetTensorA(xGM[... + loopK * quantGroupSize]);
    mm.SetTensorB(weightGM[... + loopK * quantGroupSize * N]);
    mm.SetQuantVector(weightScaleGM[... + loopK * N + tailN]);
    mm.Iterate();
    mm.GetTensorC(mmOutGM[workSpaceOffset], loopK == 0 ? 0 : 1);  // 1 == accumulate
}
```

The `loopK == 0 ? 0 : 1` argument to `GetTensorC` is the atomic-add flag: the
first block initialises the tile, every later block accumulates into it. This
maps directly onto a block-32 layout — `quantGroupSize` becomes the block, and
`quantGroupNum` the number of blocks along the reduction axis.

Set the shape once, outside the K-loop:

```cpp
mm.SetSingleShape(curSingleM, curSingleN, quantGroupSize);
```

## Staging and alignment

- `DataCopy` moves whole 32-byte blocks. For a count that is not a multiple of
  the block, use `DataCopyPad` with an explicit `DataCopyParams`, as
  `customDataCopyOut` does (`grouped_matmul_swiglu_quant.h:530-537`).
- Sub-allocate shared scratch with `TBuf::GetWithOffset<T>(size, byteOffset)`
  rather than by slicing a tensor at an element index. It states the byte offset
  explicitly, which is what the 32-byte rule is actually about.
- `AlignUp<16>(k)` / `AlignUp<32>(n)` guard NZ-format offset arithmetic; an NZ
  weight's stride is not `k * n` (`grouped_matmul_swiglu_quant.h:189-193`).
- `SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE)` on a weight tensor that is
  read once and never revisited, so it does not evict live data.

## How to verify

A dequant path is verified by numbers, never by inspection:

1. A host-tier test that checks the codec in isolation — every one of the 16
   codes, both nibble positions, and the scale decode's boundary values.
2. A kernel-level comparison against a CPU reference at a reduced geometry,
   reported as ULP distance in the output dtype.
3. For any `Cast` pair you have not used before, a case whose expected output
   differs from the input, so a silently-elided `Cast` fails rather than passes.
