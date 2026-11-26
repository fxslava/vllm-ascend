# SPDX-License-Identifier: Apache-2.0
"""Compare raw AKA diagnostics to independently parsed verified sources.

No code-category suppression is used. Raw reader findings and undecidable
domains remain visible; a zero delta is only a source-integration regression gate.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[3]
E8_LAYOUT = re.compile(r"^// (@ascend-(?:layout|reuse-group):[^\n]+)\n", re.MULTILINE)


def write_input(path: Path, text: str) -> Path:
    if not path.exists() or path.read_text() != text:
        path.write_text(text)
    return path


def e8_source_pairs(original: Path, candidate: Path, out: Path) -> dict[str, tuple[Path, Path]]:
    original_text = original.read_text()
    metadata = candidate.with_name("reader_layout.aka").read_text()
    annotations = E8_LAYOUT.findall(original_text)
    if not annotations or metadata != "\n".join(annotations) + "\n":
        raise ValueError("E8 layout sidecar differs from the independently verified reader")
    prefix = "".join(f"// {line}\n" for line in metadata.splitlines())
    annotated = write_input(out / "reader-layout-integrated.cpp", prefix + f'#include "{candidate}"\n')

    def relocate_include(match: re.Match[str]) -> str:
        include = original.parent / match[2]
        return f'{match[1]}"{include.resolve()}"' if include.is_file() else match[0]

    unannotated = re.sub(
        r'^(\s*#\s*include\s+)"([^"]+)"',
        relocate_include,
        E8_LAYOUT.sub("", original_text),
        flags=re.MULTILINE,
    )
    baseline_without_metadata = write_input(out / "reader-layout-verified-raw.cpp", unannotated)
    return {
        "e8_reader": (original, annotated),
        "e8_reader_unannotated": (baseline_without_metadata, candidate),
    }


def primary_source_report(document, inputs):
    units = [document] if len(inputs) == 1 else document.get("files", [])
    by_path = {Path(unit["source"]["path"]): unit for unit in units}
    if len(units) != len(inputs) or set(by_path) != set(inputs):
        raise RuntimeError(f"Analyzer report sources differ from the requested input set: {inputs}")
    for path, unit in by_path.items():
        if unit["source"]["had_parse_errors"] or not unit["kernels"]:
            raise RuntimeError(f"Analyzer produced an incomplete parse for {path}")
    return by_path[inputs[0]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyzer", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--reshape", type=Path, required=True)
    parser.add_argument("--e8", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", action="append", help="Run only the named case in a fresh comparison")
    args = parser.parse_args()
    for option in ("analyzer", "native", "reshape", "e8", "out"):
        setattr(args, option, getattr(args, option).resolve())
    args.out.mkdir(parents=True, exist_ok=True)
    tq = ROOT / "csrc/attention/turboquant"
    cases = {
        "decode": (
            args.native / "src/decode/turboquant_fused_decode.cpp",
            tq / "decode/native/turboquant_fused_decode.cpp",
        ),
        "vector": (
            args.native / "src/decode/vector/turboquant_vector_service.h",
            tq / "decode/native/vector/turboquant_vector_service.h",
        ),
        "softmax": (
            args.native / "src/decode/vector/routine_vector_cube_softmax.hpp",
            tq / "decode/native/vector/routine_vector_cube_softmax.hpp",
        ),
        "reshape": (
            args.reshape / "src/prefill/turboquant_reshape_batch.cpp",
            tq / "ingress/native/turboquant_reshape_batch.cpp",
        ),
        "reshape_wide": (
            args.reshape / "src/prefill/turboquant_reshape_batch.cpp",
            tq / "ingress/native/turboquant_reshape_batch_wide.cpp",
        ),
        "scatter": (
            args.native / "src/scatter/turboquant_reshape_scatter.cpp",
            tq / "ingress/native/turboquant_reshape_scatter.cpp",
        ),
        "e8_component": (args.e8 / "parallel_component.cpp", tq / "decode/e8/parallel_component.cpp"),
    }
    cases.update(e8_source_pairs(args.e8 / "e8_fused_decode.cpp", tq / "decode/e8/e8_fused_decode.cpp", args.out))
    for stage in ("reshape", "reshape_wide", "scatter"):
        nz_sources = []
        for kind, source in zip(("verified", "integrated"), cases[stage]):
            wrapper = args.out / f"{stage}-nz-{kind}.cpp"
            nz_sources.append(write_input(wrapper, f'#define TURBOQUANT_INGRESS_NZ 1\n#include "{source}"\n'))
        cases[f"{stage}_nz"] = tuple(nz_sources)
    launch_companions = {"reshape": cases["reshape_wide"][1], "reshape_nz": cases["reshape_wide_nz"][1]}
    result_path = args.out / "summary.json"
    reports = {}
    if args.case:
        unknown = set(args.case) - cases.keys()
        if unknown:
            parser.error("unknown analyzer case: " + ", ".join(sorted(unknown)))
        cases = {name: cases[name] for name in args.case}
    result_path.unlink(missing_ok=True)
    for name, pair in cases.items():
        parsed = []
        for kind, source in zip(("verified", "integrated"), pair):
            path = args.out / f"{name}-{kind}.json"
            inputs = [source]
            if kind == "integrated" and name in launch_companions:
                inputs.append(launch_companions[name])
            command = [
                "python3",
                "-m",
                "ascend_analyzer.cli",
                *map(str, inputs),
                "--chip",
                "ascend950pr",
                "--solver",
                "interval",
                "--all-functions",
                "--ignore-inline-pragmas",
                "--format",
                "json",
            ]
            with TemporaryDirectory(dir=args.out, prefix=".aka-") as directory:
                current = Path(directory) / path.name
                result = subprocess.run(
                    [*command, "-o", str(current)],
                    cwd=args.analyzer,
                    env={**os.environ, "PYTHONPATH": str(args.analyzer)},
                    capture_output=True,
                    text=True,
                )
                if result.returncode not in (0, 1) or not current.is_file():
                    raise RuntimeError(f"Analyzer did not produce a fresh report for {source}\n{result.stdout}{result.stderr}")
                data = json.loads(current.read_bytes())
                current.replace(path)
            parsed.append(primary_source_report(data, inputs))

        def fingerprints(data):
            # AKA2005 also embeds changing source locations inside its prose.
            # Preserve every other message byte and diagnostic multiplicity.
            return collections.Counter(
                (
                    d["code"],
                    d["severity"],
                    d["hardware_domain"],
                    re.sub(r"\(line \d+\)", "(line SOURCE)", d["message"]),
                )
                for d in data["diagnostics"]
            )

        before, after = map(fingerprints, parsed)
        added = after - before
        record = {
            "verified_summary": parsed[0]["summary"],
            "integrated_summary": parsed[1]["summary"],
            "new": [
                {"code": k[0], "severity": k[1], "hardware_domain": k[2], "message": k[3], "count": n}
                for k, n in sorted(added.items())
            ],
            "resolved_occurrences": sum((before - after).values()),
            "source_sha256": [hashlib.sha256(source.read_bytes()).hexdigest() for source in pair],
            "normalization": "code, severity, hardware domain, message (embedded line numbers normalized), multiplicity",
            "frontend": [
                {"functions": len(data["kernels"]), "parse_errors": data["source"]["parse_error_locations"]}
                for data in parsed
            ],
        }
        if name in launch_companions:
            record["launch_companion_sha256"] = hashlib.sha256(launch_companions[name].read_bytes()).hexdigest()
        if name.startswith("e8_reader"):
            record["reader_source_sha256"] = hashlib.sha256((tq / "decode/e8/e8_fused_decode.cpp").read_bytes()).hexdigest()
            record["layout_sidecar_sha256"] = hashlib.sha256((tq / "decode/e8/reader_layout.aka").read_bytes()).hexdigest()
        reports[name] = record
        print(json.dumps({"case": name, "new": sum(added.values()), "raw": parsed[1]["summary"]}), flush=True)
    result_path.write_text(json.dumps(reports, indent=2))
    if any(r["new"] for r in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
