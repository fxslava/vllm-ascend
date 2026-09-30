"""Thin CLI entrypoint wrapping :mod:`tools.dsv4_moe_runtime.benchmarks.offload_stress`.

Runnable both as a script (``python bench_npu_offload_stress.py --dry-run``,
from any working directory) and as a package module
(``python -m tools.dsv4_moe_runtime.bench_npu_offload_stress``).

Import hygiene: script mode adjusts ``sys.path`` with the *repository root*
only. ``tools/`` itself must never appear on ``sys.path`` -- it contains
stdlib-shadowing directories (e.g. ``tools/bisect/``, which would break
``random``'s ``from bisect import bisect``).
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # direct script execution: resolve via the repo root
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.dsv4_moe_runtime.benchmarks.offload_stress import main  # noqa: E402
else:
    from .benchmarks.offload_stress import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
