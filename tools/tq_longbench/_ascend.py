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
"""Borrow the TurboQuant cache layout from ``vllm_ascend`` without importing vLLM.

The harness drives the same operators the plugin does, so it must build the
*same* constant tables: a second implementation of
``turboquant_codec_tables`` that drifts would not raise, it would decode the
cache against the wrong centroids and return plausible wrong numbers.  Copying
them here would be exactly that second implementation.

Importing them normally is not an option either.  ``vllm_ascend/__init__.py``
reaches vLLM, and the whole point of this harness is that no part of vLLM is in
the process -- no engine, no scheduler, no config objects, and no exposure to
the version drift between a container's vLLM and the pinned baseline.

So the two modules that are already free of intra-package imports --
:mod:`vllm_ascend.attention.turboquant_layout` and
:mod:`vllm_ascend.attention.turboquant_rotation`, both of which import only
``torch`` -- are loaded **by file path**, under private module names.  Nothing
named ``vllm`` is imported, and :func:`assert_no_vllm_imported` says so.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

# Loaded under names that cannot collide with a real import of the package.
_LAYOUT_MODULE = "tq_longbench._vendored_turboquant_layout"
_ROTATION_MODULE = "tq_longbench._vendored_turboquant_rotation"

_SOURCE_ROOT_ENV = "VLLM_ASCEND_SOURCE_ROOT"

_RELATIVE = Path("vllm_ascend") / "attention"


def source_root() -> Path:
    """Return the vllm-ascend checkout holding the modules to borrow.

    ``$VLLM_ASCEND_SOURCE_ROOT`` wins, so an installed harness can point at the
    tree whose operators it is actually linked against.  Otherwise the repo this
    file sits in: ``tools/tq_longbench/_ascend.py`` -> two parents up.
    """
    override = os.environ.get(_SOURCE_ROOT_ENV)
    if override:
        root = Path(override).expanduser()
        if not (root / _RELATIVE / "turboquant_layout.py").is_file():
            raise FileNotFoundError(
                f"{_SOURCE_ROOT_ENV}={root} does not look like a vllm-ascend checkout: "
                f"{root / _RELATIVE / 'turboquant_layout.py'} is missing"
            )
        return root
    return Path(__file__).resolve().parents[2]


def _load(module_name: str, file_name: str) -> ModuleType:
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    path = source_root() / _RELATIVE / file_name
    if not path.is_file():
        raise FileNotFoundError(
            f"cannot borrow {file_name}: {path} does not exist. Set {_SOURCE_ROOT_ENV} to a vllm-ascend checkout."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build a module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec so a traceback inside names the module properly.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module


def turboquant_layout() -> ModuleType:
    """The cache geometry and codec tables: the shared source of truth."""
    return _load(_LAYOUT_MODULE, "turboquant_layout.py")


def turboquant_rotation() -> ModuleType:
    """``Pi`` itself: the sign draw, the Walsh-Hadamard transform, and the o_proj fold."""
    return _load(_ROTATION_MODULE, "turboquant_rotation.py")


def assert_no_vllm_imported() -> None:
    """Raise if anything pulled vLLM into this process.

    The harness exists to measure the operators with no framework around them.
    A stray ``import vllm`` would not break a run, it would quietly reintroduce
    the thing being excluded -- so the check is an assertion, made where it is
    cheap, rather than a comment claiming the property.
    """
    leaked = sorted(name for name in sys.modules if name == "vllm" or name.startswith("vllm."))
    if leaked:
        raise RuntimeError(
            "tq_longbench runs with no vLLM in the process, but these modules are loaded: "
            + ", ".join(leaked[:8])
            + ("..." if len(leaked) > 8 else "")
        )
