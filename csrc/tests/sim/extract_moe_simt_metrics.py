#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
"""Extract modeled clocks, never simulator wall time. Missing counters stay null."""
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path


def extract(root, elements):
    if elements <= 0:
        raise ValueError("elements must be positive")
    summaries = list(root.rglob("summary.json"))
    if len(summaries) != 1:
        raise ValueError(f"expected one kernel summary under {root}, found {len(summaries)}")
    summary_path = summaries[0]
    summary = json.loads(summary_path.read_text())
    clocks = summary["kernel_info"]["kernel_total_clocks"]
    vf_path = summary_path.parent / "kernel_launch_0_vf_report.json"
    vf_report = json.loads(vf_path.read_text()) if vf_path.exists() else {}
    vfs = vf_report.get("vfs", [])
    counts = Counter()
    traces = list(summary_path.parent.glob("core_*/trace_core*.json*"))
    for trace in traces:
        opener = gzip.open if trace.suffix == ".gz" else open
        with opener(trace, "rt") as handle:
            for event in json.load(handle).get("traceEvents", []):
                if event.get("ph") == "X":
                    counts[event.get("name", "unknown")] += 1
    return {
        "summary": str(summary_path), "elements": elements,
        "kernel_info": summary["kernel_info"],
        "vf_kernel_info": vf_report.get("kernel"),
        "cache_report": summary.get("cache"),
        "bandwidth_report": summary.get("bandwidth"),
        "pipeline_overlap": summary.get("pipeline_overlap"),
        "scalar_instructions": summary.get("scalar_instructions"),
        "modeled_cycles": clocks, "cycles_per_element_end_to_end": clocks / elements,
        "pipeline_utilization": summary.get("pipe_utilization"),
        "vector_instruction_report": summary.get("aiv_vector_instructions"),
        "trace_event_counts": dict(counts), "vector_functions": vfs,
        "simt_vfs": [vf for vf in vfs if vf.get("type") == "simt"],
        "exclusive_stall_cycles": None, "data_cache_hit_rate": None,
        "gm_bandwidth_bytes_per_cycle": None, "isolated_mode_switch_cycles": None,
        "notes": ["Pipeline utilizations overlap: do not add them to derive total cycles.",
                  "VF duration includes work and is not isolated mode-switch latency.",
                  "Logical byte counts are not measured bus traffic."]}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--elements", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.elements <= 0:
        parser.error("elements must be positive")
    args.output.write_text(json.dumps(extract(args.report, args.elements), indent=2) + "\n")


if __name__ == "__main__":
    main()
