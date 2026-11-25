# SPDX-License-Identifier: Apache-2.0
"""Syntax-check the Torch bindings and standalone host tiling using header stubs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TQ = ROOT / "csrc/attention/turboquant"


def run() -> None:
    torch_root = Path(torch.__file__).resolve().parent
    flags = [
        "g++",
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I",
        str(ROOT),
        "-I",
        str(HERE / "stubs"),
        "-isystem",
        str(torch_root / "include"),
        "-isystem",
        str(torch_root / "include/torch/csrc/api/include"),
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
    ]
    sources = [
        TQ / "bindings/torch_bindings.cpp",
        TQ / "bindings/torch_ingress.cpp",
        TQ / "bindings/torch_decode.cpp",
        TQ / "common/turboquant_tiling.cpp",
        HERE / "abi.cpp",
    ]
    for source in sources:
        subprocess.run([*flags, "-fsyntax-only", str(source)], check=True)
    print(json.dumps({"torch": torch.__version__, "syntax_translation_units": len(sources)}))


if __name__ == "__main__":
    run()
