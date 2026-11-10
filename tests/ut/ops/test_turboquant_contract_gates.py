# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project
"""Host-only gates for the TurboQuant layout and synchronization contracts.

These replace the single ``test_turboquant_kernel_contracts`` gate. They are the
two device-free legs of the audited kernel tree, and they are toolkit-free on
purpose: the layout contract, the host tiling and the kernels' host-visible
geometry and nothing else, so a build plus a run is a couple of seconds with any
C++17 compiler and needs no CANN and no NPU.

``test_contract_gates`` covers the fused decode's UB arena ``static_assert``\\ s,
the host tiling's dispatch odometer and descriptors against the arithmetic they
replace, the launch guards' admissions and refusals, the mode-2 channel map, the
shared online-softmax phase order, and a marked-graph token-balance audit of the
prefill's seamless chunk pipeline.

``test_layout_bridge`` checks that the NZ ingress address function and an
independently re-derived decode reader name the same byte and the same nibble for
every logical element, that both the NZ and the RFC 3.1 strip maps are bijections
onto their allocations, that the two contracts' co-byte pairings are structurally
incompatible, and that both scale planes are bijections.

Run without the NPU unit-test fixtures::

    pytest --noconftest tests/ut/ops/test_turboquant_contract_gates.py
"""

import shutil
import subprocess
from pathlib import Path

import pytest

GATES = ("test_contract_gates", "test_layout_bridge")

# `turboquant_reshape_batch.cpp` is included by test_contract_gates.cpp: its
# device body sits behind `#if defined(__CCE_AICORE__)`, so a host compiler sees
# only the arena constants and their static_asserts, which is exactly what the
# gate is for. The kernel's own `-Werror` budget is not spent on CANN headers
# here because none are reachable on this path.
COMPILE_FLAGS = ("-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror")

# `build_decode_tiling` is a definition in a translation unit now, not an `inline`
# in a device-side header, so the gate that exercises it has to link that unit.
# That is the point of the move: the host tiling is ordinary host C++ and the
# gate compiles the very same object the launcher does.
TILING_TU = "csrc/attention/turboquant/op_host/turboquant_tiling.cpp"
EXTRA_SOURCES = {"test_contract_gates": (TILING_TU,)}


@pytest.fixture(scope="module")
def gate_binaries(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the host C++ TurboQuant contract gates")
    root = Path(__file__).resolve().parents[3]
    out = tmp_path_factory.mktemp("turboquant_gates")
    binaries = {}
    for gate in GATES:
        source = root / "tests" / "ut" / "ops" / "turboquant" / f"{gate}.cpp"
        extra = [str(root / name) for name in EXTRA_SOURCES.get(gate, ())]
        binary = out / gate
        result = subprocess.run(
            [compiler, *COMPILE_FLAGS, "-I", str(root), str(source), *extra, "-o", str(binary)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{gate} failed to build:\n{result.stdout}{result.stderr}"
        binaries[gate] = binary
    return binaries


@pytest.mark.parametrize("gate", GATES)
def test_turboquant_contract_gate(gate_binaries, gate):
    result = subprocess.run([str(gate_binaries[gate])], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
