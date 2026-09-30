#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Golden data generator and comparator for the dsv4_moe_expert kernel.

The kernel under test fuses one DeepSeek-V4 routed expert:

    gate = x @ dequant(w1)^T        up = x @ dequant(w3)^T
    activated = silu(gate) * up
    down = activated @ dequant(w2)^T

Weights are packed FP4 (E2M1, two nibbles per byte) with block-32 E8M0 scales.
This module is the single source of numeric truth for Gate B: every bit-level
helper here mirrors one in op_kernel/dsv4_moe_expert.cpp.

Contract (kernel and golden must agree exactly):
  * packed FP4: byte i holds logical element 2i in the LOW nibble and element
    2i+1 in the HIGH nibble;
  * E8M0 scale byte b decodes to 2^(b-127); b=0 is the fp32 subnormal 2^-127
    (bit pattern 0x00400000), b=0xFF is NaN, per OCP MX;
  * accumulation is fp32, one multiply-add per element, reduction dimension
    walked in ascending order;
  * outputs are bf16, rounded round-to-nearest-even, specials truncated.

Usage:
    gen_golden.py gen --out-dir DIR [--hidden 256] [--inter 128] [--seed 0]
    gen_golden.py cmp --golden DIR/golden.bin --actual DIR/actual.bin
                      [--meta DIR/meta.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

FP4_BLOCK = 32
FP4_PER_BYTE = 2

# DeepSeek-V4 architectural constant: the gate activation is clamped
# symmetrically to +/- 10.0 before SwiGLU. The kernel carries the same value in
# its tiling struct; the two must agree or they compute different functions.
SWIGLU_LIMIT = np.float32(10.0)

# E2M1 nibble decode: bit3 sign, bits2..1 exponent (bias 1), bit0 mantissa.
E2M1_TABLE = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)

# Output buffers in kernel argument order. Gate B compares all four.
OUTPUT_ORDER = ("gate_out", "up_out", "activated", "down_out")
# Input buffers in kernel argument order.
INPUT_ORDER = ("x", "w1", "w2", "w3", "w1_scale", "w2_scale", "w3_scale")


# --------------------------------------------------------------------------
# bit-level helpers (mirrors of the kernel's scalar conversions)
# --------------------------------------------------------------------------
def e8m0_to_scale(exponent_bytes: np.ndarray) -> np.ndarray:
    """Decode E8M0 bytes to exact fp32 powers of two. Mirrors E8M0ToScale()."""
    b = np.asarray(exponent_bytes, dtype=np.uint32)
    bits = np.where(b == 0, np.uint32(0x00400000), (b << np.uint32(23)))
    out = bits.astype(np.uint32).view(np.float32).copy()
    out[b == 0xFF] = np.float32(np.nan)
    return out


def float_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """fp32 -> bf16 bits, round-to-nearest-even. Mirrors FloatToBf16Bits()."""
    f = np.ascontiguousarray(values, dtype=np.float32)
    bits = f.view(np.uint32)
    is_special = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = ((bits + rounding) >> np.uint32(16)).astype(np.uint16)
    truncated = (bits >> np.uint32(16)).astype(np.uint16)
    return np.where(is_special, truncated, rounded).astype(np.uint16)


def bf16_bits_to_float(bits: np.ndarray) -> np.ndarray:
    """bf16 bits -> fp32 (exact widening). Mirrors Bf16BitsToFloat()."""
    b = np.asarray(bits, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return b.view(np.float32).copy()


def unpack_fp4(packed: np.ndarray, cols: int) -> np.ndarray:
    """[rows, cols/2] packed bytes -> [rows, cols] fp32 E2M1 values."""
    low = E2M1_TABLE[(packed & np.uint8(0x0F)).astype(np.intp)]
    high = E2M1_TABLE[(packed >> np.uint8(4)).astype(np.intp)]
    out = np.empty((packed.shape[0], cols), dtype=np.float32)
    out[:, 0::2] = low
    out[:, 1::2] = high
    return out


def dequantize(packed: np.ndarray, scales: np.ndarray, cols: int) -> np.ndarray:
    """Packed FP4 + block-32 E8M0 -> fp32 weight matrix [rows, cols]."""
    weights = unpack_fp4(packed, cols)
    block_scales = e8m0_to_scale(scales)  # [rows, cols/32]
    expanded = np.repeat(block_scales, FP4_BLOCK, axis=1)  # [rows, cols]
    return (weights * expanded).astype(np.float32)


# --------------------------------------------------------------------------
# reference kernel
# --------------------------------------------------------------------------
def project(x_f32: np.ndarray, packed: np.ndarray, scales: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """out[r] = sum_c x[c] * (E2M1(w[r,c]) * E8M0(scale[r, c/32])).

    The reduction is done in ascending column order with fp32 accumulation, so
    it matches the kernel's summation sequence element for element.
    """
    weights = dequantize(packed, scales, cols)
    products = (x_f32[np.newaxis, :].astype(np.float32) * weights).astype(np.float32)
    out = np.zeros(rows, dtype=np.float32)
    for c in range(cols):
        out = (out + products[:, c]).astype(np.float32)
    return out


def reference_expert(
    x_bits: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    w3: np.ndarray,
    w1_scale: np.ndarray,
    w2_scale: np.ndarray,
    w3_scale: np.ndarray,
    hidden: int,
    inter: int,
) -> dict[str, np.ndarray]:
    """Full expert pipeline, returning bf16-bit views of the four outputs."""
    x_f32 = bf16_bits_to_float(x_bits)

    gate = project(x_f32, w1, w1_scale, inter, hidden)
    up = project(x_f32, w3, w3_scale, inter, hidden)

    # DeepSeek-V4 clamps the gate symmetrically to +/- swiglu_limit before the
    # activation. This is part of the model definition, not an overflow guard:
    # at +/-10 the exponential stays far inside fp32 range, so the inf path an
    # unclamped implementation relies on never arises.
    #
    # The clamp goes to a separate array: `gate_out` is the RAW projection, so
    # that output keeps meaning "the gate GEMM's result".
    gate_clamped = np.clip(gate, -SWIGLU_LIMIT, SWIGLU_LIMIT).astype(np.float32)
    # silu(g) * u, evaluated as (g * sigmoid(g)) * u to match the kernel.
    neg_exp = np.exp((-gate_clamped).astype(np.float32)).astype(np.float32)
    sigmoid = (np.float32(1.0) / (np.float32(1.0) + neg_exp)).astype(np.float32)
    activated = ((gate_clamped * sigmoid).astype(np.float32) * up).astype(np.float32)

    down = project(activated, w2, w2_scale, hidden, inter)

    return {
        "gate_out": float_to_bf16_bits(gate),
        "up_out": float_to_bf16_bits(up),
        "activated": float_to_bf16_bits(activated),
        "down_out": float_to_bf16_bits(down),
    }


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def _scale_bytes(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """E8M0 bytes in a healthy 2^-6 .. 2^+6 window.

    Gate B measures the kernel's arithmetic, not its behaviour on MX specials;
    0x00 (subnormal) and 0xFF (NaN) are deliberately excluded so a genuine
    arithmetic regression cannot hide behind a NaN comparison.
    """
    return rng.integers(127 - 6, 127 + 7, size=shape, dtype=np.int64).astype(np.uint8)


def generate(out_dir: str, hidden: int, inter: int, seed: int) -> int:
    if hidden % (2 * FP4_BLOCK) or inter % (2 * FP4_BLOCK):
        print("hidden and inter must be multiples of 64", file=sys.stderr)
        return 2

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # x is generated in bf16 space so the kernel's widening is exact.
    x_bits = float_to_bf16_bits(rng.standard_normal(hidden, dtype=np.float32))
    w1 = rng.integers(0, 256, size=(inter, hidden // FP4_PER_BYTE), dtype=np.int64).astype(np.uint8)
    w3 = rng.integers(0, 256, size=(inter, hidden // FP4_PER_BYTE), dtype=np.int64).astype(np.uint8)
    w2 = rng.integers(0, 256, size=(hidden, inter // FP4_PER_BYTE), dtype=np.int64).astype(np.uint8)
    w1_scale = _scale_bytes(rng, (inter, hidden // FP4_BLOCK))
    w3_scale = _scale_bytes(rng, (inter, hidden // FP4_BLOCK))
    w2_scale = _scale_bytes(rng, (hidden, inter // FP4_BLOCK))

    inputs = {
        "x": x_bits,
        "w1": w1,
        "w2": w2,
        "w3": w3,
        "w1_scale": w1_scale,
        "w2_scale": w2_scale,
        "w3_scale": w3_scale,
    }
    outputs = reference_expert(x_bits, w1, w2, w3, w1_scale, w2_scale, w3_scale, hidden, inter)

    meta = {
        "hidden": hidden,
        "inter": inter,
        "block": FP4_BLOCK,
        "tiling": {"hiddenSize": hidden, "interSize": inter, "blockSize": FP4_BLOCK},
        "seed": seed,
        "inputs": [],
        "outputs": [],
    }

    def _pack(names: list[str], source: dict[str, np.ndarray], key: str, path: str) -> None:
        offset = 0
        with open(path, "wb") as handle:
            for name in names:
                raw = np.ascontiguousarray(source[name]).tobytes()
                handle.write(raw)
                meta[key].append(
                    {
                        "name": name,
                        "offset": offset,
                        "bytes": len(raw),
                        "shape": list(np.shape(source[name])),
                        "dtype": "bf16_bits" if source[name].dtype == np.uint16 else "uint8",
                    }
                )
                offset += len(raw)

    _pack(list(INPUT_ORDER), inputs, "inputs", os.path.join(out_dir, "input.bin"))
    _pack(list(OUTPUT_ORDER), outputs, "outputs", os.path.join(out_dir, "golden.bin"))

    with open(os.path.join(out_dir, "meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2)

    # Self-check: a second evaluation must be bit-identical (guards against any
    # non-determinism sneaking into the reference itself).
    again = reference_expert(x_bits, w1, w2, w3, w1_scale, w2_scale, w3_scale, hidden, inter)
    for name in OUTPUT_ORDER:
        if not np.array_equal(again[name], outputs[name]):
            print("SELF-CHECK FAILED for {}".format(name), file=sys.stderr)
            return 1

    print(f"wrote {out_dir}: hidden={hidden} inter={inter} block={FP4_BLOCK} seed={seed}")
    for entry in meta["inputs"] + meta["outputs"]:
        print(f"  {entry['name']:<10} offset={entry['offset']:<8} bytes={entry['bytes']:<8} shape={entry['shape']}")
    print("SELF-CHECK OK (reference is deterministic)")
    return 0


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
def _ulp_distance(a_bits: np.ndarray, b_bits: np.ndarray) -> np.ndarray:
    """Distance in bf16 ULPs, using the monotone sign-magnitude -> ordinal map."""

    def ordinal(bits: np.ndarray) -> np.ndarray:
        b = bits.astype(np.int32)
        neg = (b & 0x8000) != 0
        return np.where(neg, 0x8000 - (b & 0x7FFF), b + 0x8000).astype(np.int64)

    return np.abs(ordinal(a_bits) - ordinal(b_bits))


def compare(golden_path: str, actual_path: str, meta_path: str) -> int:
    with open(meta_path) as handle:
        meta = json.load(handle)

    golden = np.fromfile(golden_path, dtype=np.uint8)
    actual = np.fromfile(actual_path, dtype=np.uint8)
    if golden.size != actual.size:
        print(f"SIZE MISMATCH: golden={golden.size} bytes actual={actual.size} bytes", file=sys.stderr)
        return 1

    worst_ulp = 0
    worst_rate = 0.0
    worst_rel = 0.0
    failed = False

    print(f"{'buffer':<12} {'elems':>10} {'FpDiff':>10} {'RateDiff':>12} {'MaxRelErr':>12}")
    print("-" * 60)
    for entry in meta["outputs"]:
        start, end = entry["offset"], entry["offset"] + entry["bytes"]
        g_bits = golden[start:end].view(np.uint16)
        a_bits = actual[start:end].view(np.uint16)

        ulp = _ulp_distance(a_bits, g_bits)
        fp_diff = int(ulp.max()) if ulp.size else 0
        rate_diff = float((ulp > 0).sum()) / float(ulp.size) if ulp.size else 0.0

        g_val = bf16_bits_to_float(g_bits)
        a_val = bf16_bits_to_float(a_bits)
        denom = np.maximum(np.abs(g_val), np.float32(1e-30))
        max_rel = float(np.max(np.abs(a_val - g_val) / denom)) if g_val.size else 0.0

        worst_ulp = max(worst_ulp, fp_diff)
        worst_rate = max(worst_rate, rate_diff)
        worst_rel = max(worst_rel, max_rel)
        if fp_diff > 2 or rate_diff > 1e-2:
            failed = True

        print(f"{entry['name']:<12} {ulp.size:>10} {fp_diff:>10} {rate_diff:>12.6f} {max_rel:>12.3e}")

    print("-" * 60)
    print(f"FpDiff   (max bf16 ULP distance, pass <= 2)     : {worst_ulp}")
    print("RateDiff (fraction of elements differing, <=1e-2): {:.6f}".format(worst_rate))
    print("MaxRelErr(max relative error)                   : {:.3e}".format(worst_rel))
    print(f"VERDICT: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("gen", help="write input.bin/golden.bin/meta.json")
    gen.add_argument("--out-dir", required=True)
    gen.add_argument("--hidden", type=int, default=256)
    gen.add_argument("--inter", type=int, default=128)
    gen.add_argument("--seed", type=int, default=0)

    cmp_parser = sub.add_parser("cmp", help="compare kernel output against the golden")
    cmp_parser.add_argument("--golden", required=True)
    cmp_parser.add_argument("--actual", required=True)
    cmp_parser.add_argument("--meta", default=None)

    args = parser.parse_args(argv)
    if args.command == "gen":
        return generate(args.out_dir, args.hidden, args.inter, args.seed)

    meta = args.meta or os.path.join(os.path.dirname(os.path.abspath(args.golden)), "meta.json")
    return compare(args.golden, args.actual, meta)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
