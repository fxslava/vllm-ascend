"""Thin CLI entrypoint wrapping :mod:`dsv4_moe_runtime.benchmarks.offload_stress`.

Runnable both as a script (``python bench_npu_offload_stress.py --dry-run``,
from any working directory) and as a package module
(``python -m tools.dsv4_moe_runtime.bench_npu_offload_stress``).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # direct script execution: make the package importable
    # NOTE: append, not prepend -- tools/ contains stdlib-shadowing directories
    # (e.g. tools/bisect/), so it must come AFTER the stdlib on sys.path.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dsv4_moe_runtime.benchmarks.offload_stress import main  # noqa: E402
else:
    from .benchmarks.offload_stress import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
