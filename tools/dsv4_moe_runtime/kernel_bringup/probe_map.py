#!/usr/bin/env python3
"""Map the kernel's effective gather codes against the packed nibbles."""
import struct

import numpy as np

d = open("/tmp/dsv4_gateb/actual.bin", "rb").read()
inp = open("/tmp/dsv4_gateb/input.bin", "rb").read()
got = np.array(struct.unpack("<64f", d[0:256]), dtype=np.float32)  # products probe (gate slot)

x = np.frombuffer(inp[:512], dtype="<u2").astype(np.uint32) << np.uint32(16)
xf = x.view(np.float32)
w1 = np.frombuffer(inp[512 : 512 + 16384], dtype=np.uint8)
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
lo = (w1[:128] & 0xF).astype(np.int32)
lo_vals = E2M1[w1[:128] & 0xF]
want = (xf[:128] * lo_vals).astype(np.float32)

print("got [0:12]  :", [round(v, 4) for v in got[:12]])
print("want[0:12]  :", [round(v, 4) for v in want[:12]])
print("codes lo    :", lo[:12].tolist())
# effective code per element: value / x  -> nearest E2M1 magnitude
n = min(len(got), 128)
with np.errstate(divide="ignore", invalid="ignore"):
    eff = np.where(xf[:n] != 0, got[:n] / xf[:n], np.nan)
eff_codes = []
for v in eff[:16]:
    m = [k for k in range(16) if abs(float(E2M1[k]) - (v if not np.isnan(v) else 1e9)) < 1e-6]
    eff_codes.append(m[0] if m else None)
print("eff codes   :", eff_codes)
