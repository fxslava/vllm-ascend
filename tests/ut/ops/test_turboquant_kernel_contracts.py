# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Host-only gates for the shared device layout and synchronization contracts.

Run without the NPU unit-test fixtures:
pytest --noconftest tests/ut/ops/test_turboquant_kernel_contracts.py
"""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def kernel_contract_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the host C++ kernel contract tests")
    source = Path(__file__).with_suffix(".cpp")
    root = source.parents[3]
    binary = tmp_path_factory.mktemp("turboquant_contracts") / "contracts"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(root), str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("contract", ["wave", "decode"])
def test_turboquant_kernel_contracts(kernel_contract_binary, contract):
    result = subprocess.run([str(kernel_contract_binary), contract], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
