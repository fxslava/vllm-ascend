#!/usr/bin/env python3
"""Four-way probe dump: read the output slots as raw fp32."""
import struct

d = open("/tmp/dsv4_gateb/actual.bin", "rb").read()
# actual.bin: gate_out [0:256), up_out [256:512), activated [512:768), down_out [768:1280)
import sys
if len(sys.argv) > 1 and sys.argv[1] == "down":
    vals = struct.unpack("<128f", d[768:1280])
    print("down slot:", [round(v, 4) for v in vals[:16]])
elif len(sys.argv) > 1 and sys.argv[1] == "ctor":
    print("spacingIdx@ctor =", struct.unpack("<16I", d[0:64]))
    print("lut16@ctor      =", [round(v, 4) for v in struct.unpack("<16f", d[256:320])])
elif len(sys.argv) > 1 and sys.argv[1] == "cast":
    acc = struct.unpack("<32f", d[768:896])
    row = struct.unpack("<32H", d[256:320])
    print("accRow  :", [round(v, 4) for v in acc[:12]])
    print("rowBf16 :", [hex(v) for v in row[:12]])
elif len(sys.argv) > 1 and sys.argv[1] == "idx":
    vals = struct.unpack("<64I", d[768 : 768 + 256])
    print("spacingIdx[0:16] =", vals[:16])
else:
    for name, o, n in [("products", 0, 64), ("partRaw", 256, 64), ("merge0", 512, 64), ("accRow", 768, 128)]:
        vals = struct.unpack(f"<{n}f", d[o : o + 4 * n])
        print(name, [round(v, 4) for v in vals[:12]])
