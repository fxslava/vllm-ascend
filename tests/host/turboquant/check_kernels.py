# SPDX-License-Identifier: Apache-2.0
"""CANN syntax-only gate; emits no device objects and links no plugin."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", help="Recheck one source stem and retain other recorded results")
    parser.add_argument("--core", type=Path, help="Also verify the standalone core and its legacy launch declarations")
    parser.add_argument("--strip", action="store_true", help="Check the standalone strip writer pair")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    compiler = args.sdk / "tools/bisheng_compiler/bin/bisheng"
    root = args.core / "src/turboquant" if args.core else ROOT / "csrc/attention/turboquant"
    sources = [
        root / "ingress/native/turboquant_reshape_batch.cpp",
        root / "ingress/native/turboquant_reshape_batch_wide.cpp",
        root / "ingress/native/turboquant_reshape_scatter.cpp",
        root / "decode/native/turboquant_fused_decode.cpp",
    ]
    if args.core:
        sources += [root / "decode/e8/e8_fused_decode.cpp", root / "decode/e8/parallel_component.cpp"]
    else:
        sources += [root / "bindings/asc_e8_reader.cpp", root / "bindings/asc_e8_component.cpp"]
    if args.strip:
        if not args.core:
            parser.error("--strip requires --core")
        sources = sources[:3]
    result_path = args.out / "syntax.json"
    results = json.loads(result_path.read_text()) if args.source and result_path.exists() else []
    if args.source:
        sources = [s for s in sources if s.stem == args.source]
        if not sources:
            parser.error("unknown source stem")
        results = [r for r in results if Path(r["source"]).stem != args.source]
    for source in sources:
        for role in ("host", "device"):
            log = args.out / f"{source.stem}-{role}.log"
            command = [
                str(compiler),
                "-x",
                "asc",
                "--npu-arch=dav-3510",
                "--cce-host-only" if role == "host" else "--cce-aicore-only",
                "-fsyntax-only",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-DASCEND_MICRO_CANN_LAUNCH",
                "-DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1",
            ]
            if not args.strip:
                command += ["-DTURBOQUANT_INGRESS_NZ=1"]
            # System SDK headers cannot spend the project's diagnostic budget.
            for directory in (args.sdk / "x86_64-linux/asc").rglob("*"):
                if directory.is_dir():
                    command += ["-isystem", str(directory)]
            command += ["-isystem", str(args.sdk / "include")]
            if source.stem.startswith("turboquant_"):
                if args.core:
                    command += ["-DTURBOQUANT_LEGACY_CANN_LAUNCH=1"]
            else:
                command += [
                    "-DENABLE_CV_COMM_VIA_SSBUF=1",
                    "-fno-fast-math",
                    "-ffp-contract=off",
                ]
            command += [str(source)]
            with log.open("w") as stream:
                result = subprocess.run(
                    command,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, "ASCEND_HOME_PATH": str(args.sdk)},
                )
            record = {
                "source": str(source.relative_to(args.core or ROOT)),
                "role": role,
                "exit": result.returncode,
                "command": command,
                "log": str(log),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            results.append(record)
            print(json.dumps({k: record[k] for k in ("source", "role", "exit")}), flush=True)
    result_path.write_text(json.dumps(results, indent=2))
    if any(r["exit"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
