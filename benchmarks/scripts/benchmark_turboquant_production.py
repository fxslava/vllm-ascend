# SPDX-License-Identifier: Apache-2.0
"""Run the standalone TurboQuant production matrix from a synchronized core checkout."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

REPOSITORY_PARENT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--core-root", type=Path, help="Synchronized turboquant-core checkout")
    args, forwarded = parser.parse_known_args()
    roots = (
        (args.core_root,)
        if args.core_root
        else (
            REPOSITORY_PARENT / "turboquant-core",
            REPOSITORY_PARENT / "Ascend" / "turboquant-core",
        )
    )
    runners = (root / "benchmarks" / "benchmark_turboquant_production.py" for root in roots)
    runner = next((candidate for candidate in runners if candidate.is_file()), None)
    if runner is None:
        if "-h" in forwarded or "--help" in forwarded:
            parser.print_help()
            return
        parser.exit(2, "Pass --core-root with the path to the synchronized turboquant-core checkout.\n")
    sys.argv = [str(runner), *forwarded]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
