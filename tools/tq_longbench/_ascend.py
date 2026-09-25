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

The operators themselves come from a shared library. Without the full extension,
:func:`load_turboquant_library` opens the standalone one that
``tools/tq_longbench/build_turboquant_ops.py`` builds -- a library of kernels and
op registrations, which imports no Python from ``vllm_ascend`` at all.
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

# Loaded under names that cannot collide with a real import of the package.
_LAYOUT_MODULE = "tq_longbench._vendored_turboquant_layout"
_ROTATION_MODULE = "tq_longbench._vendored_turboquant_rotation"

#: ``turboquant_sink`` is the one borrowed module with intra-package imports of its own
#: -- it reaches ``turboquant_layout`` and ``turboquant_rotation``, which are already
#: borrowed here. It therefore cannot be loaded by file path alone: the names it imports
#: have to resolve first. :func:`turboquant_sink` stands the two of them up under their
#: real dotted names inside shell packages whose ``__init__`` never runs, execs the file,
#: and takes the shells back down. Nothing named ``vllm`` is imported, and the module
#: keeps direct references to what it imported, so removing the shells afterwards leaves
#: it working and leaves no ``vllm_ascend`` in ``sys.modules`` for anything else to find.
_SINK_MODULE = "tq_longbench._vendored_turboquant_sink"

_SHELL_PACKAGES = ("vllm_ascend", "vllm_ascend.attention")

_SOURCE_ROOT_ENV = "VLLM_ASCEND_SOURCE_ROOT"

#: A standalone TurboQuant library to load, or a directory holding one. When set it
#: is the only candidate: a run asked for one build must not quietly get another.
TURBOQUANT_LIB_ENV = "TURBOQUANT_LIB_PATH"

#: What ``tools/tq_longbench/build_turboquant_ops.py`` installs.
TURBOQUANT_LIB_NAME = "libvllm_turboquant_cube.so"

#: The Ascend C kernels the binding launches, installed beside it.
TURBOQUANT_KERNELS_LIB_NAME = "libvllm_turboquant_cube_kernels.so"

#: A build tree to take the standalone library from, searched after
#: :data:`TURBOQUANT_LIB_ENV` and before the installed location. Unlike
#: ``$TURBOQUANT_LIB_PATH`` this *adds* candidates rather than replacing them, so a host
#: that exports it for the loader's kernel search can export it here too without
#: narrowing the search to one directory.
#:
#: Note what a ``csrc/tests`` build tree does and does not hold. Its
#: ``lib/libvllm_ascend_turboquant.so`` is the Ascend C kernel library and nothing else:
#: ``csrc/tests`` is configured without Python, PyTorch or torch_npu on purpose, so that
#: file links no ``libc10`` and registers no ``torch.ops`` schema at all -- handing it to
#: ``torch.ops.load_library`` succeeds and registers nothing. The operators come only from
#: :data:`TURBOQUANT_LIB_NAME`, which only ``build_turboquant_ops.py`` produces. No
#: ``setup.py`` and no ``pip install`` anywhere on that path.
TURBOQUANT_BUILD_DIR_ENV = "ASCEND_TQ_BUILD_DIR"

#: The conventional name of the RUN_MODE=npu ``csrc/tests`` tree, relative to the
#: checkout. Searched so that a host which staged the binding beside those binaries finds
#: it without setting anything.
TURBOQUANT_TEST_BUILD_DIR = Path("build") / "csrc-tests-npu"

#: The operators whose schema is checked before a launch, and the argument each has to
#: carry. ``lse`` is the softmax-statistics out-tensor both decodes grew for the
#: uncompressed attention sinks (TURBOQUANT_TESTS.md 13.37). A library that predates it
#: registers the same operator *names* one argument short, and the dispatcher then refuses
#: the call with "expected at most N argument(s)" from inside a decode step -- minutes into
#: a run, naming neither the stale file nor the feature that needs the new one.
TURBOQUANT_SCHEMA_PROBES = {
    "npu_turboquant_paged_attention": "lse",
    "npu_turboquant_cube_decode": "lse",
}

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


def shell_package(name: str) -> ModuleType:
    """A package object with an empty ``__path__`` whose ``__init__`` never runs.

    ``vllm_ascend/__init__.py`` reaches vLLM and ``tests/ut/conftest.py`` reaches the
    plugin; neither is wanted, and neither is needed to resolve a submodule that is about
    to be loaded by file path anyway.
    """
    module = sys.modules.get(name)
    if module is not None:
        return module
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    parent, _, leaf = name.rpartition(".")
    if parent:
        setattr(shell_package(parent), leaf, module)
    return module


def turboquant_sink() -> ModuleType:
    """The uncompressed attention sinks: the side-car plane and the softmax merge.

    See :data:`_SINK_MODULE` for why this one needs more than a file-path load. The shells
    are removed again on the way out, whether the exec succeeded or not, so a later
    ``import vllm_ascend`` anywhere cannot be answered by a hollow package this left behind.
    """
    cached = sys.modules.get(_SINK_MODULE)
    if cached is not None:
        return cached

    borrowed = {
        "vllm_ascend.attention.turboquant_layout": turboquant_layout(),
        "vllm_ascend.attention.turboquant_rotation": turboquant_rotation(),
    }
    # Only what this call plants comes back down again: cpu_reference stands up the same
    # shells for the CPU stand-ins and keeps them, and removing another component's entry
    # would be a failure it has no way to see coming.
    planted = [name for name in (*_SHELL_PACKAGES, *borrowed) if name not in sys.modules]
    attention = shell_package("vllm_ascend.attention")
    for name, borrowed_module in borrowed.items():
        sys.modules.setdefault(name, borrowed_module)
        setattr(attention, name.rpartition(".")[2], borrowed_module)
    try:
        module = _load(_SINK_MODULE, "turboquant_sink.py")
    finally:
        for name in reversed(planted):
            sys.modules.pop(name, None)
    assert_no_vllm_imported()
    return module


@dataclass(frozen=True)
class LibraryLoad:
    """Whether the operators are registered now, and what was done to get there."""

    registered: bool
    detail: str


def _library_in(directory: Path) -> list[Path]:
    """``TURBOQUANT_LIB_NAME`` in ``directory`` and in its ``lib/``, in that order.

    A CMake tree puts its shared libraries under ``lib/`` while a staged directory
    usually holds them flat, and a caller pointing at a build tree should not have to
    know which.
    """
    return [directory / TURBOQUANT_LIB_NAME, directory / "lib" / TURBOQUANT_LIB_NAME]


def turboquant_library_candidates() -> list[Path]:
    """Where a standalone TurboQuant library may be, in the order they are tried.

    ``$TURBOQUANT_LIB_PATH`` (a file, or a directory holding
    :data:`TURBOQUANT_LIB_NAME`, flat or under ``lib/``) replaces the search rather than
    joining it: a run that asked for one build must not quietly get another.

    Otherwise, in order: ``$ASCEND_TQ_BUILD_DIR``, ``tools/tq_longbench/lib/`` where the
    build script installs, the checkout's ``build/csrc-tests-npu/``, and the checkout's
    ``build/``. Duplicates are dropped so that a build dir which is already one of the
    defaults is not reported twice.
    """
    override = os.environ.get(TURBOQUANT_LIB_ENV)
    if override:
        path = Path(override).expanduser()
        return _library_in(path) if path.is_dir() else [path]

    candidates: list[Path] = []
    build_dir = os.environ.get(TURBOQUANT_BUILD_DIR_ENV)
    if build_dir:
        path = Path(build_dir).expanduser()
        candidates += _library_in(path) if path.is_dir() or not path.suffix else [path]
    candidates.append(Path(__file__).resolve().parent / "lib" / TURBOQUANT_LIB_NAME)
    candidates += _library_in(source_root() / TURBOQUANT_TEST_BUILD_DIR)
    candidates.append(source_root() / "build" / TURBOQUANT_LIB_NAME)

    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def turboquant_schema(name: str):
    """The registered schema of ``_C_ascend::<name>``, or ``None`` if nothing answers.

    The schema lives on the operator's *overload*, not on the ``OpOverloadPacket`` that
    ``torch.ops`` hands out: a packet carries no ``_schema`` at all, so reading one off it
    is a check that can only ever pass. ``default`` is the overload a schema with no
    overload name registers, which is all of these.

    ``None`` means "nothing here could answer" -- the operator is not registered, or what
    stands in for it is a test double whose attributes answer to every name -- and never
    "this library is stale", so a caller has to treat it as a pass.
    """
    import torch

    namespace = getattr(torch.ops, "_C_ascend", None)
    if namespace is None:
        return None
    operator = getattr(namespace, name, None)
    if operator is None:
        return None
    schema = getattr(getattr(operator, "default", operator), "_schema", None)
    return schema if isinstance(schema, torch._C.FunctionSchema) else None


def loaded_library_paths(needle: str) -> list[str]:
    """Every mapped file whose path contains ``needle``, from ``/proc/self/maps``.

    This is what names the library that actually won the registration. The dispatcher
    records no provenance for a schema, so without this a stale operator is a fact with
    no file attached to it, and the operator cannot tell you which of several candidate
    builds it came from. Silent where there is no procfs, which is not Linux's problem to
    have but is worth not crashing over.
    """
    try:
        with open("/proc/self/maps", encoding="utf-8") as maps:
            lines = maps.read().splitlines()
    except OSError:
        return []
    paths = {
        line.split()[-1]
        for line in lines
        if len(line.split()) >= 6 and line.split()[-1].startswith("/") and needle in line.split()[-1]
    }
    return sorted(paths)


def turboquant_mapped_libraries() -> list[str]:
    """Every mapped file that could plausibly have registered the TurboQuant operators.

    Both spellings, because the registration can come from the full extension (whose file
    is named after the ``_C_ascend`` namespace) or from a standalone binding (named after
    the kernels). An empty list means no mapped file is named after either -- not that the
    operators are unregistered, and not that procfs is missing; the two are indistinguishable
    from here and neither is worth guessing at in a message.
    """
    seen = dict.fromkeys(loaded_library_paths("_C_ascend") + loaded_library_paths("turboquant"))
    return list(seen)


def turboquant_schema_report() -> list[str]:
    """One line per probed operator: its argument count, and whether it carries the probe.

    For a banner and for the preflight. Operators this build never registered are
    reported as such rather than skipped, because "the Cube decode is absent" and "the
    Cube decode is stale" are different problems with the same symptom at the call site.
    """
    lines = []
    for name, required in TURBOQUANT_SCHEMA_PROBES.items():
        schema = turboquant_schema(name)
        if schema is None:
            lines.append(f"{name}: not registered")
            continue
        arguments = [argument.name for argument in schema.arguments]
        carries = "yes" if required in arguments else f"NO -- '{required}' is missing"
        lines.append(f"{name}: {len(arguments)} args, {required} = {carries}")
    return lines


def assert_turboquant_schema_is_fresh() -> None:
    """Refuse a registration that predates the ``lse`` out-tensor, naming the file.

    Raised here rather than left to the call because the dispatcher's own message --
    "expected at most 17 argument(s) but received 18" -- arrives from inside a decode
    step, after the weights have loaded, and names neither the file that registered the
    old schema nor the argument that grew.

    **A stale winner cannot be corrected from Python.** Operator registration is global
    and permanent for the life of the process: once a library has ``def``-ed
    ``_C_ascend::npu_turboquant_cube_decode``, loading another that defines the same name
    raises rather than replacing it. So the only fix is to stop the stale library being
    loaded at all, and the only useful thing to do here is to say which one it was.
    """
    stale = []
    for name, required in TURBOQUANT_SCHEMA_PROBES.items():
        schema = turboquant_schema(name)
        if schema is None:
            continue
        if required not in [argument.name for argument in schema.arguments]:
            stale.append((name, schema))
    if not stale:
        return

    mapped = turboquant_mapped_libraries()
    where = "\n".join(f"  loaded: {path}" for path in mapped) or "  (no mapped file is named after either)"
    details = "\n".join(f"  {name} registered as: {schema}" for name, schema in stale)
    raise RuntimeError(
        "the TurboQuant operators in this process predate the 'lse' softmax-statistics out-tensor, so a "
        f"decode would be refused mid-run with an argument count.\n{details}\n{where}\n"
        "Operator registration is global and permanent: a second library defining the same names cannot "
        "replace these, so loading a fresh one will not help. Stop the stale library from being loaded -- it "
        "is one of the files above -- and let the loader find a current "
        f"{TURBOQUANT_LIB_NAME}. Build one with tools/tq_longbench/build_turboquant_ops.py (no setup.py and "
        f"no pip install), and point ${TURBOQUANT_LIB_ENV} or ${TURBOQUANT_BUILD_DIR_ENV} at it. Searched: "
        + ", ".join(str(path) for path in turboquant_library_candidates())
    )


def load_turboquant_library(is_registered: Callable[[], bool]) -> LibraryLoad:
    """Register the TurboQuant operators from a standalone library, if they are missing.

    Nothing is loaded when ``is_registered()`` already holds -- the full extension,
    the CPU stand-ins or an earlier load got there first, and loading a second
    library would redefine the same ``_C_ascend`` schemas. Otherwise the first
    existing candidate is opened with ``torch.ops.load_library`` and the check is
    repeated, so a library that loads but lacks the operator is reported as such.
    The detail names the path and the loader's own error, which is what a
    failure on the NPU host needs to be diagnosed from a log.

    When the operators were already there, the detail names the mapped files that could
    have registered them. "Already registered" on its own is the least useful thing this
    can say on a host with more than one build of the extension installed, which is
    exactly where it matters: a stale one that got in first cannot be replaced, and the
    first question is then which file it was. :func:`assert_turboquant_schema_is_fresh` is
    what turns that into a refusal.
    """
    if is_registered():
        mapped = turboquant_mapped_libraries()
        where = f" (mapped: {', '.join(mapped)})" if mapped else ""
        return LibraryLoad(True, f"already registered{where}")
    candidates = turboquant_library_candidates()
    present = [path for path in candidates if path.is_file()]
    if not present:
        where = f"${TURBOQUANT_LIB_ENV}" if os.environ.get(TURBOQUANT_LIB_ENV) else "the default locations"
        searched = ", ".join(str(path) for path in candidates)
        return LibraryLoad(
            False,
            f"no standalone TurboQuant library at {where} ({searched}); build one with "
            "tools/tq_longbench/build_turboquant_ops.py",
        )

    import torch

    library = present[0]
    _import_torch_npu()
    try:
        _preload_kernel_library(library)
    except OSError as error:
        return LibraryLoad(False, f"preloading the kernels beside {library} failed: {error}")
    try:
        torch.ops.load_library(str(library))
    except OSError as error:
        return LibraryLoad(False, f"torch.ops.load_library({library}) failed: {error}")
    if not is_registered():
        return LibraryLoad(False, f"loaded {library}, but it does not register the operator")
    return LibraryLoad(True, f"loaded {library}")


def _preload_kernel_library(library: Path) -> Path | None:
    """Load the kernel library beside ``library`` first, globally, if it is there.

    The binding finds it through its ``$ORIGIN`` RUNPATH -- when that survived the
    build: a CMake 3.22 ``-Wl,-rpath,$ORIGIN`` link option lands as a literal
    ``$$ORIGIN``, and the load fails with "cannot open shared object file" while
    both files sit side by side. Once a library with the kernels' SONAME is
    loaded, the dynamic linker satisfies the binding's NEEDED entry with it, so
    this makes the load independent of how the binding was linked. After
    torch_npu, whose ACL symbols the kernel library leaves undefined.
    """
    kernels = library.parent / TURBOQUANT_KERNELS_LIB_NAME
    if not kernels.is_file():
        return None
    ctypes.CDLL(str(kernels), mode=ctypes.RTLD_GLOBAL)
    return kernels


def _import_torch_npu() -> None:
    """The library links torch_npu, and its NPU kernels expect the runtime initialised."""
    if "torch_npu" in sys.modules or importlib.util.find_spec("torch_npu") is None:
        return
    import torch_npu  # noqa: F401


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
