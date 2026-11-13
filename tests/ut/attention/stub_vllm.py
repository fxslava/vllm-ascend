# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stand up the narrowest vllm stubs that let the device-free TurboQuant tests run.

``test_turboquant_{op_schemas,dispatch_seam,strip_cache,codec,
kernel_cache_layout,layout,nz_ingress,profile}.py`` need nothing but ``torch``:
they check layout arithmetic, the kernel/golden dispatch seam, the in-code
profiler's bookkeeping, and the ``ops.def`` schemas in
``csrc/attention/turboquant/torch_bindings.cpp`` against the Python stubs that
call them. The only reason they need a vllm install at all is
``turboquant_v1.py``'s own import block, and the only reason they need an
Ascend host is ``tests/ut/conftest.py``.

This module removes both reasons, so the seam can be checked on a developer
machine with no NPU, no CANN and no vllm. It is NOT a substitute for the real
suite: anything that actually exercises vLLM plumbing or a device belongs in
``tests/ut`` or ``tests/e2e`` as usual and will not work here.

Run it through :mod:`run_turboquant_device_free`, which is the supported entry
point. Importing this module has no effect; call :func:`install`.

Why stubs rather than the pinned vendor container: the container tracks one vllm
version, and a branch based on a newer one cannot even import
``tests/ut/conftest.py`` there -- so on those branches this is the only way to
run these checks at all.
"""

from __future__ import annotations

import enum
import importlib.machinery
import importlib.util
import logging
import pathlib
import sys
import types

# tests/ut/attention/stub_vllm.py -> the checkout root.
REPO = pathlib.Path(__file__).resolve().parents[3]


def vllm_is_importable() -> bool:
    """Whether the real vllm is present, in which case stubbing is wrong."""
    try:
        return importlib.util.find_spec("vllm") is not None
    except (ImportError, ValueError):
        return False


def _module(name: str, path: pathlib.Path | None = None) -> types.ModuleType:
    """Put a module (or, with ``path``, a package) into ``sys.modules``.

    A package gets a ``__path__`` and a matching ``__spec__``, so the normal
    import machinery keeps working underneath it. That is the point of using a
    real directory here: ``vllm_ascend.attention.turboquant_layout`` and
    ``...turboquant_v1`` are then found and imported from the checkout,
    rather than loaded from somewhere else by hand. Keeping them at their true
    paths matters -- ``test_turboquant_op_schemas.py`` locates
    ``torch_bindings.cpp`` by walking up from
    ``turboquant_v1.__file__``, and silently skips every one of its tests
    if that walk does not land on the checkout.
    """
    module = types.ModuleType(name)
    if path is not None:
        module.__path__ = [str(path)]
        module.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)
        module.__spec__.submodule_search_locations = [str(path)]
    sys.modules[name] = module
    return module


def install() -> None:
    """Install the stubs. Refuses to run where the real vllm is importable.

    The refusal is the safety property that lets this module live beside the
    tests it serves: a CI run, or any developer with vllm installed, gets the
    real thing, and nothing here can shadow it.
    """
    if vllm_is_importable():
        raise RuntimeError(
            "vllm is importable here, so these stubs must not be installed -- "
            "run the tests directly with pytest instead."
        )

    # torch_npu is imported by turboquant_v1 for its side effects; nothing
    # in these tests calls into it.
    _module("torch_npu")

    _module("vllm")
    _module("vllm.logger").logger = logging.getLogger("vllm")

    _module("vllm.v1")
    _module("vllm.v1.attention")
    backend = _module("vllm.v1.attention.backend")

    class AttentionLayer:
        """Only ever a type annotation in the code under test."""

    class AttentionType:
        ENCODER_ONLY = "encoder_only"
        DECODER = "decoder"

    backend.AttentionLayer = AttentionLayer
    backend.AttentionType = AttentionType

    _module("vllm.v1.attention.backends")
    registry = _module("vllm.v1.attention.backends.registry")

    # A real Enum, because vLLM indexes it by name -- AttentionBackendEnum[
    # backend.get_name()] -- and members must answer get_class() with whatever
    # register_backend bound to them. A plain class with class attributes passes
    # import but fails both, which is what
    # test_backend_name_is_the_enum_member_it_registers_as checks.
    backend_enum = enum.Enum("AttentionBackendEnum", {"TURBOQUANT": "TURBOQUANT"})
    bound: dict[object, type] = {}
    backend_enum.get_class = lambda self: bound[self]  # type: ignore[attr-defined]

    def register_backend(member: object, *args: object, **kwargs: object):
        def bind(cls: type) -> type:
            bound[member] = cls
            return cls

        return bind

    registry.AttentionBackendEnum = backend_enum
    registry.register_backend = register_backend

    # vllm_ascend and vllm_ascend.attention as bare packages over the real
    # directories: this bypasses vllm_ascend/__init__.py, whose triton shims and
    # logger would pull vllm back in, while leaving every module underneath to be
    # imported normally from the checkout.
    _module("vllm_ascend", REPO / "vllm_ascend")
    _module("vllm_ascend.attention", REPO / "vllm_ascend" / "attention")

    attention_v1 = _module("vllm_ascend.attention.attention_v1")

    class AscendAttentionBackend:
        """Base of the TurboQuant backend; the tests only read its subclass."""

    class AscendAttentionBackendImpl:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class AscendAttentionMetadataBuilder:
        pass

    class AscendAttentionState:
        PrefillNoCache = "PrefillNoCache"
        PrefillCacheHit = "PrefillCacheHit"
        ChunkedPrefill = "ChunkedPrefill"
        DecodeOnly = "DecodeOnly"
        SpecDecoding = "SpecDecoding"

    class AscendMetadata:
        pass

    attention_v1.AscendAttentionBackend = AscendAttentionBackend
    attention_v1.AscendAttentionBackendImpl = AscendAttentionBackendImpl
    attention_v1.AscendAttentionMetadataBuilder = AscendAttentionMetadataBuilder
    attention_v1.AscendAttentionState = AscendAttentionState
    attention_v1.AscendMetadata = AscendMetadata

    # turboquant_layout is pure os/typing/torch, so it is NOT stubbed -- the real
    # one is what the layout tests are about. Only this one helper is needed from
    # attention.utils, whose own imports reach deep into vllm.
    _module("vllm_ascend.attention.utils").notify_kv_cache_written = lambda *a, **kw: None
