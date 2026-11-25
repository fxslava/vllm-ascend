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
    parser.add_argument("--core", type=Path, help="Also usable against the standalone core's legacy source layout")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    compiler = args.sdk / "tools/bisheng_compiler/bin/bisheng"
    root = ROOT / "csrc/attention/turboquant"
    sources = [
        root / "op_kernel" / f"turboquant_{name}.cpp" for name in ("reshape_batch", "reshape_scatter", "fused_decode")
    ]
    sources += [root / "e8/asc_reader.cpp", root / "e8/asc_component.cpp"]
    if args.core:
        root = args.core
        sources = [
            root / f"src/{part}/turboquant_{name}.cpp"
            for part, name in (("prefill", "reshape_batch"), ("scatter", "reshape_scatter"), ("decode", "fused_decode"))
        ]
        sources += [
            root / "components/e8_parallel_decode/e8_fused_decode.cpp",
            root / "components/e8_parallel_decode/parallel_component.cpp",
        ]
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
                "-DENABLE_CV_COMM_VIA_SSBUF=1",
                "-DTURBOQUANT_INGRESS_NZ=1",
                "-DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1",
            ]
            # System SDK headers cannot spend the project's diagnostic budget.
            for directory in (args.sdk / "x86_64-linux/asc").rglob("*"):
                if directory.is_dir():
                    command += ["-isystem", str(directory)]
            command += ["-isystem", str(args.sdk / "include")]
            if source.stem.startswith("turboquant_"):
                header = root / "op_adapter/turboquant_torch_adpt.h"
                if args.core:
                    header = root / "include/turboquant/turboquant_launch.h"
                command += ["-include", str(header)]
            else:
                e8_root = root / ("components/e8_parallel_decode" if args.core else "e8")
                command += [
                    "-include",
                    str(e8_root / "e8_decode.hpp"),
                    "-include",
                    str(e8_root / "e8_component.hpp"),
                    "-include",
                    "kernel_operator.h",
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
