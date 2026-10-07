"""Package-wide Python 3.8/3.9 import-compatibility guards.

Every module must carry ``from __future__ import annotations`` (so PEP 604
unions and builtin-generic annotations stay unevaluated strings on <3.10),
parse under the 3.8 grammar, and importing the base package must not eagerly
pull in the benchmark tooling.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def _package_modules():
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def test_every_module_parses_under_python38_grammar():
    for path in _package_modules():
        ast.parse(path.read_text(encoding="utf-8"), feature_version=(3, 8))


def test_every_module_starts_with_future_annotations_import():
    for path in _package_modules():
        body = ast.parse(path.read_text(encoding="utf-8")).body
        assert body, f"{path}: empty module"
        index = 0
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            index = 1  # module docstring may precede the future import
        future = body[index] if index < len(body) else None
        ok = isinstance(future, ast.ImportFrom) and future.module == "__future__" and [
            alias.name for alias in future.names
        ] == ["annotations"]
        assert ok, f"{path}: first import must be 'from __future__ import annotations'"


def _run_python(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=600)


def test_base_package_import_does_not_load_benchmarks():
    result = _run_python(
        "import sys; import tools.dsv4_moe_runtime; "
        "loaded = [m for m in sys.modules if m.startswith('tools.dsv4_moe_runtime.benchmarks')]; "
        "print(loaded)"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_lazy_benchmark_reexport_still_resolves():
    result = _run_python(
        "import sys; "
        "from tools.dsv4_moe_runtime import OffloadStressHarness; "
        "assert 'tools.dsv4_moe_runtime.benchmarks.offload_stress' in sys.modules; "
        "print('ok')"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
