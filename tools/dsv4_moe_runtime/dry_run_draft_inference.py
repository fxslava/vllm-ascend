"""Thin CLI entrypoint wrapping :mod:`dsv4_moe_runtime.draft_inference.dry_run`.

Runnable as a script from any working directory or via
``python -m tools.dsv4_moe_runtime.dry_run_draft_inference``.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):  # direct script execution: make the package importable
    # NOTE: append, never prepend -- tools/ contains stdlib-shadowing
    # directories (e.g. tools/bisect/).
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from dsv4_moe_runtime.draft_inference.dry_run import main  # noqa: E402
else:
    from .draft_inference.dry_run import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
