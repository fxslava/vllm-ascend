#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Serve the TurboQuant operators from the CPU, so the harness runs with no device.

``tests/ut/attention/turboquant_cpu_ops.py`` already implements what the
kernels compute, enforces what the adapter refuses, and registers itself into
``torch.ops._C_ascend`` for the duration of a ``with`` block.  Reaching it means
the harness can be exercised end to end -- paging, the tie point, the write and
read-back of the packed planes -- on a laptop, and held to exact attention
rather than to "nothing raised".

Borrowing it takes more than an import, because that module's two dependencies
(``turboquant_rotation`` and one constant tuple out of ``turboquant_v1``) sit
inside a package whose ``__init__`` reaches vLLM.  So shell packages are
registered for the path to it, ``turboquant_v1`` is stood up as a shim over the
constants :mod:`tq_longbench._ascend` already borrowed, and the module itself is
loaded by file path.  Nothing named ``vllm`` is imported.

Only the vector (AIV) decode has a CPU kernel.  The Cube decode gets **meta**
kernels only -- shape checks, no arithmetic -- because what that launch computes
is verified on the camodel, not here.  Asking the Cube backend for numbers under
this context manager therefore returns whatever the output buffer held; the
harness's tests use ``turboquant_aiv``, and :func:`cube_meta_only` says why.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

from tq_longbench._ascend import (
    assert_no_vllm_imported,
    shell_package,
    source_root,
    turboquant_layout,
    turboquant_rotation,
)

_CPU_OPS_MODULE = "tests.ut.attention.turboquant_cpu_ops"

_SHELL_PACKAGES = ("vllm_ascend", "vllm_ascend.attention", "tests", "tests.ut", "tests.ut.attention")

_RELATIVE = Path("tests") / "ut" / "attention" / "turboquant_cpu_ops.py"


def _install_dependencies() -> None:
    """Put ``turboquant_rotation`` and a ``turboquant_v1`` shim where the stand-ins look."""
    for name in _SHELL_PACKAGES:
        shell_package(name)

    attention = sys.modules["vllm_ascend.attention"]
    rotation = turboquant_rotation()
    sys.modules["vllm_ascend.attention.turboquant_rotation"] = rotation
    attention.turboquant_rotation = rotation  # type: ignore[attr-defined]

    # The stand-ins want exactly one name out of turboquant_v1, and the real
    # module reaches the vLLM attention backend to get it. The constant itself
    # lives in turboquant_layout, which is already borrowed, so the shim hands
    # over the same tuple rather than a copy of it.
    shim = ModuleType("vllm_ascend.attention.turboquant_v1")
    shim.TURBOQUANT_LLOYD_MAX_THRESHOLDS = turboquant_layout().TURBOQUANT_LLOYD_MAX_THRESHOLDS  # type: ignore[attr-defined]
    sys.modules["vllm_ascend.attention.turboquant_v1"] = shim
    attention.turboquant_v1 = shim  # type: ignore[attr-defined]


def cpu_ops_module() -> ModuleType:
    """Load ``turboquant_cpu_ops`` by file path, with its dependencies stood up first."""
    cached = sys.modules.get(_CPU_OPS_MODULE)
    if cached is not None:
        return cached
    _install_dependencies()
    path = source_root() / _RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"the CPU operator stand-ins are missing: {path}")
    spec = importlib.util.spec_from_file_location(_CPU_OPS_MODULE, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build a module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_CPU_OPS_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[_CPU_OPS_MODULE]
        raise
    assert_no_vllm_imported()
    return module


@contextlib.contextmanager
def cpu_turboquant_ops(vector_cores: int | None = None) -> Iterator[ModuleType]:
    """Serve ``torch.ops._C_ascend.npu_turboquant_*`` from the CPU inside the block.

    Yields the stand-in module itself, because a caller checking the harness
    against exact attention needs its ``dequantize`` and its centroid table to
    say what the cache actually holds.
    """
    module = cpu_ops_module()
    cores = module.CPU_VECTOR_CORES if vector_cores is None else vector_cores
    with module.turboquant_cpu_ops(cores):
        yield module


@contextlib.contextmanager
def cube_meta_only() -> Iterator[ModuleType]:
    """Define the Cube operators with **meta** kernels: shapes checked, nothing computed.

    Enough to prove the harness hands the Cube decode a self-consistent call --
    the head counts, the one-table-row-per-token block table, the fp32
    ``query_rot`` -- which is the part that can be wrong on a host. The numbers
    come from the camodel gate, ``test_sim_950pr_turboquant_fused``.
    """
    module = cpu_ops_module()
    with module.turboquant_cube_meta_ops():
        yield module
