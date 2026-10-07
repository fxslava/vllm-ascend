#!/usr/bin/env python3
"""Probe analyzer: decode actual.bin down_out slot as raw fp32 products and
compare against the expected x*E2M1 (scale-less) values from input.bin."""

from __future__ import annotations

import json
import struct

import numpy as np

meta = json.load(open("/tmp/dsv4_gateb/meta.json"))
data = open("/tmp/dsv4_gateb/actual.bin", "rb").read()
off = next(e for e in meta["outputs"] if e["name"] == "down_out")
off, n = off["offset"], off["bytes"]
vals = struct.unpack("<128f", data[off : off + n])
print("products[0:16] bits+vals:")
for i in range(16):
    print(f"  [{i}] {vals[i]:.6g}  (bits {struct.unpack('<I', struct.pack('<f', vals[i]))[0]:#010x})")

inp = open("/tmp/dsv4_gateb/input.bin", "rb").read()
x = np.frombuffer(inp[:512], dtype="<u2").astype(np.uint32) << np.uint32(16)
xf = x.view(np.float32)
w1 = np.frombuffer(inp[512 : 512 + 16384], dtype=np.uint8)
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
row0 = w1[:128]  # w1 row 0, hidden=256 -> 128 packed bytes
lo = E2M1[row0 & 0xF]
hi = E2M1[row0 >> 4]
exp = np.empty(256, dtype=np.float32)
exp[0::2] = lo
exp[1::2] = hi
want = (xf * exp).astype(np.float32)
got = np.array(vals[:256], dtype=np.float32)
print("match x*E2M1 row0:", np.array_equal(got, want))
print("kernel [0:8] :", [round(v, 4) for v in got[:8]])
print("expect[0:8]  :", [round(v, 4) for v in want[:8]])
# hypothesis search over a few candidate mappings
scales = np.frombuffer(inp[512+16384*2:512+16384*2+128], dtype=np.uint8)  # w1_scale [128 rows, 8]
sc0 = np.float32(2.0) ** (np.int32(scales[0]) - 127)
hi_vals = np.empty(256, dtype=np.float32)
hi_vals[0::2] = E2M1[row0 >> 4]
hi_vals[1::2] = E2M1[row0 >> 4]
cands = {
    "x*hi": xf * hi_vals,
    "x*hi*sc": (xf * hi_vals * sc0).astype(np.float32),
    "x*lo*sc": (xf * exp * sc0).astype(np.float32),
    "x*lo/2": (xf * exp * np.float32(0.5)).astype(np.float32),
    "x alone": xf.astype(np.float32),
    "x/2": (xf * np.float32(0.5)).astype(np.float32),
}
for name, c in cands.items():
    c32 = np.asarray(c, dtype=np.float32)
    if np.array_equal(got[:256], c32):
        print("MATCH:", name)
    elif np.array_equal(got[:256], -c32):
        print("MATCH: -", name)
print("x[0:8]        =", [round(v, 4) for v in xf[:8]])
print("hi vals[0:8]  =", [round(v, 4) for v in hi_vals[:8]])
print("scale0        =", sc0)
# compare the raw vLo codes against every nibble stream of the packed bytes
got_lo = vals[:128]
wrow = w1[:128]
nib = np.empty(256, dtype=np.int32)
nib[0::2] = wrow & 0xF
nib[1::2] = wrow >> 4
E2M1V = {0: 0.0, 1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 5: 3.0, 6: 4.0, 7: 6.0,
         8: -0.0, 9: -0.5, 10: -1.0, 11: -1.5, 12: -2.0, 13: -3.0, 14: -4.0, 15: -6.0}
code_of = {v: k for k, v in E2M1V.items()}
kernel_codes = []
for v in got_lo[:16]:
    kernel_codes.append(code_of.get(abs(v), f"?{v}"))
print("kernel vLo codes[0:16]:", kernel_codes)
print("packed lo nibbles [0:16]:", [int(v) for v in (wrow & 0xF)[:16]])
print("packed hi nibbles [0:16]:", [int(v) for v in (wrow >> 4)[:16]])
print("interleaved nibbles[0:16]:", [int(v) for v in nib[:16]])
