# SPDX-License-Identifier: Apache-2.0
"""Compare raw AKA diagnostics to independently parsed verified sources.

No code-category suppression is used. Raw reader findings and undecidable
domains remain visible; a zero delta is only a source-integration regression gate.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--reshape", type=Path, required=True)
    parser.add_argument("--e8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reuse", action="store_true", help="Reuse reports newer than their input files")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tq = ROOT / "csrc/attention/turboquant"
    cases = {
        "decode": (
            args.native / "src/decode/turboquant_fused_decode.cpp",
            tq / "op_kernel/turboquant_fused_decode.cpp",
        ),
        "vector": (
            args.native / "src/decode/vector/turboquant_vector_service.h",
            tq / "op_kernel/vector/turboquant_vector_service.h",
        ),
        "softmax": (
            args.native / "src/decode/vector/routine_vector_cube_softmax.hpp",
            tq / "op_kernel/vector/routine_vector_cube_softmax.hpp",
        ),
        "reshape": (
            args.reshape / "src/prefill/turboquant_reshape_batch.cpp",
            tq / "op_kernel/turboquant_reshape_batch.cpp",
        ),
        "scatter": (
            args.native / "src/scatter/turboquant_reshape_scatter.cpp",
            tq / "op_kernel/turboquant_reshape_scatter.cpp",
        ),
        "e8_reader": (args.e8 / "e8_fused_decode.cpp", tq / "e8/e8_fused_decode.cpp"),
        "e8_component": (args.e8 / "parallel_component.cpp", tq / "e8/parallel_component.cpp"),
    }
    nz_sources = []
    for kind, source in zip(("verified", "integrated"), cases["reshape"]):
        wrapper = args.out / f"reshape-nz-{kind}.cpp"
        wrapper.write_text(f'#define TURBOQUANT_INGRESS_NZ 1\n#include "{source}"\n')
        nz_sources.append(wrapper)
    cases["reshape_nz"] = tuple(nz_sources)
    reports = {}
    for name, pair in cases.items():
        parsed = []
        for kind, source in zip(("verified", "integrated"), pair):
            path = args.out / f"{name}-{kind}.json"
            command = [
                "python3",
                "-m",
                "ascend_analyzer.cli",
                str(source),
                "--chip",
                "ascend950pr",
                "--solver",
                "interval",
                "--all-functions",
                "--format",
                "json",
                "-o",
                str(path),
            ]
            reuse = False
            if args.reuse and path.exists():
                previous = json.loads(path.read_text())
                generated = datetime.datetime.fromisoformat(previous["tool"]["generated_at"]).timestamp()
                reuse = previous["source"]["path"] == str(source) and source.stat().st_mtime < generated
            if not reuse:
                result = subprocess.run(
                    command,
                    cwd=args.analyzer,
                    env={**os.environ, "PYTHONPATH": str(args.analyzer)},
                    capture_output=True,
                    text=True,
                )
                if result.returncode not in (0, 1):
                    raise RuntimeError(result.stdout + result.stderr)
            data = json.loads(path.read_text())
            parsed.append(data)

        def fingerprints(data):
            # AKA2005 also embeds changing source locations inside its prose.
            # Preserve every other message byte and diagnostic multiplicity.
            return collections.Counter(
                (d["code"], d["severity"], re.sub(r"\(line \d+\)", "(line SOURCE)", d["message"]))
                for d in data["diagnostics"]
            )

        before, after = map(fingerprints, parsed)
        added = after - before
        record = {
            "verified_summary": parsed[0]["summary"],
            "integrated_summary": parsed[1]["summary"],
            "new": [{"code": k[0], "severity": k[1], "message": k[2], "count": n} for k, n in sorted(added.items())],
            "resolved_occurrences": sum((before - after).values()),
            "source_sha256": [hashlib.sha256(source.read_bytes()).hexdigest() for source in pair],
            "normalization": "location and embedded '(line N)' only; no categories suppressed",
        }
        reports[name] = record
        print(json.dumps({"case": name, "new": sum(added.values()), "raw": parsed[1]["summary"]}), flush=True)
    (args.out / "summary.json").write_text(json.dumps(reports, indent=2))
    if any(r["new"] for r in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
