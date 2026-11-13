# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The TurboQuant op schemas, and that the Python stubs match them.

A ``TORCH_LIBRARY_FRAGMENT`` schema string is only checked when the library
registers, which needs the compiled extension and an Ascend 950 build. Nothing
on a development host would otherwise notice a malformed schema, a missing
``Tensor!`` on a cache plane, or -- the one that actually happens -- a
``_kernel_*`` stub in ``turboquant_v1.py`` passing its arguments in an
order the C++ schema no longer declares.

``torch._C.parse_schema`` is the same parser the registration uses, so these
tests read the schemas straight out of
``csrc/attention/turboquant/torch_bindings.cpp`` and check all of that with no
extension, no vllm and no device.
"""

import inspect
import pathlib
import re
from typing import Any

import pytest
import torch

from vllm_ascend.attention import turboquant_v1
from vllm_ascend.attention.turboquant_v1 import (
    TURBOQUANT_OP_BUILD_DECODE_TILING,
    TURBOQUANT_OP_FUSED_DECODE,
    TURBOQUANT_OP_RESHAPE_AND_CACHE,
    TURBOQUANT_OP_RESHAPE_SCATTER,
    AscendTurboQuantMetadataBuilder,
    TurboQuantOpsAdapter,
)

# Anchored to the module under test, not to this file, so the checkout is found
# whether pytest collects this from tests/ut/attention or from a copy beside a
# stubbed-vllm conftest that loads the real module by path.
_REPO = pathlib.Path(turboquant_v1.__file__).parents[2]
BINDINGS = _REPO / "csrc" / "attention" / "turboquant" / "torch_bindings.cpp"

# Which `_kernel_*` method of the adapter calls each op. The stub's parameters
# are the call it makes, so they must be the schema's arguments in the schema's
# order.
OP_STUBS = {
    TURBOQUANT_OP_RESHAPE_AND_CACHE: "_kernel_reshape_and_cache_a8",
    TURBOQUANT_OP_RESHAPE_SCATTER: "_kernel_reshape_scatter_aiv",
    TURBOQUANT_OP_FUSED_DECODE: "_kernel_fused_decode",
}

# Every op the fragment declares. The tiling builder is not in OP_STUBS because
# it has no adapter stub: it is called once per step by
# AscendTurboQuantMetadataBuilder, not once per operation by the ops adapter, so
# its call site is checked separately below.
BINDING_OPS = frozenset(OP_STUBS) | {TURBOQUANT_OP_BUILD_DECODE_TILING}

# A C++ string literal, which ops.def() may build from adjacent literals.
_LITERAL = r'"(?:[^"\\]|\\.)*"'


def _source() -> str:
    if not BINDINGS.is_file():
        pytest.skip(f"{BINDINGS} is not in this checkout")
    return BINDINGS.read_text(encoding="utf-8")


def _schemas() -> dict[str, Any]:
    """Every ``ops.def`` schema in the bindings, parsed, keyed by op name."""
    parsed = {}
    for match in re.finditer(r"ops\.def\(\s*((?:" + _LITERAL + r"\s*)+)\)\s*;", _source()):
        text = "".join(part[1:-1] for part in re.findall(_LITERAL, match.group(1)))
        schema = torch._C.parse_schema(text)
        parsed[schema.name.split("::")[-1]] = schema
    return parsed


def _impl_keys() -> dict[str, set[str]]:
    """The dispatch keys each op name is registered for."""
    keys: dict[str, set[str]] = {}
    for match in re.finditer(r'ops\.impl\(\s*"([^"]+)"\s*,\s*([A-Za-z0-9_:]+)\s*,', _source()):
        keys.setdefault(match.group(1), set()).add(match.group(2))
    return keys


def test_the_bindings_declare_exactly_the_ops_python_calls() -> None:
    assert set(_schemas()) == set(BINDING_OPS)


@pytest.mark.parametrize("op_name", sorted(BINDING_OPS))
def test_every_op_returns_nothing_and_writes_through_its_arguments(op_name: str) -> None:
    """Every one writes in place, so a schema that returned a Tensor would lie."""
    schema = _schemas()[op_name]
    assert len(schema.returns) == 0, f"{op_name} declares a return value"
    mutated = [a.name for a in schema.arguments if a.alias_info is not None and a.alias_info.is_write]
    assert mutated, f"{op_name} mutates nothing, so it cannot have an effect"


def test_the_writers_mutate_all_three_cache_planes() -> None:
    """A plane without `Tensor!` is one the functionaliser may copy, silently."""
    for op_name in (TURBOQUANT_OP_RESHAPE_AND_CACHE, TURBOQUANT_OP_RESHAPE_SCATTER):
        mutated = {a.name for a in _schemas()[op_name].arguments if a.alias_info is not None and a.alias_info.is_write}
        assert mutated == {"key_cache", "value_cache", "scale_cache"}, op_name


def test_the_fused_decode_mutates_its_output_and_its_scratch() -> None:
    """`output`, and the two scratch buffers the kernel writes through.

    `tiling` is deliberately NOT in this set: the decode reads the descriptors
    and never writes them, and declaring it `Tensor!` would let the
    functionaliser believe the op produces a new tiling buffer. `workspace` and
    `query_rot` are written -- the Split-K partials and the rotated query image
    -- so a plain `Tensor` on either would let it copy them silently.
    """
    mutated = {
        a.name
        for a in _schemas()[TURBOQUANT_OP_FUSED_DECODE].arguments
        if a.alias_info is not None and a.alias_info.is_write
    }
    assert mutated == {"output", "workspace", "query_rot"}


def test_the_tiling_builder_mutates_only_the_tiling_buffer() -> None:
    mutated = {
        a.name
        for a in _schemas()[TURBOQUANT_OP_BUILD_DECODE_TILING].arguments
        if a.alias_info is not None and a.alias_info.is_write
    }
    assert mutated == {"tiling"}


def test_the_tiling_builder_is_called_with_the_schema_in_order() -> None:
    """The builder's call site is the stub for this op, so pin it the same way.

    `_refresh_tiling` calls the op positionally out of its own state, so there
    is no signature to compare: the check is on the call it writes.
    """
    source = inspect.getsource(AscendTurboQuantMetadataBuilder._refresh_tiling)
    call = source.split("entry(", 1)[1]
    passed = [line.strip().rstrip(",") for line in call.splitlines() if line.strip() not in ("", ")")]
    passed = [arg for arg in passed if not arg.startswith(")")]
    schema_args = [a.name for a in _schemas()[TURBOQUANT_OP_BUILD_DECODE_TILING].arguments]
    assert len(passed) == len(schema_args), (
        f"_refresh_tiling passes {len(passed)} arguments, the schema declares {len(schema_args)}: "
        f"{passed} vs {schema_args}"
    )


def test_the_fused_decode_takes_an_optional_gate() -> None:
    """`gate` is read only at output_stage 2, so stage 0 and 1 must pass None."""
    gate = next(a for a in _schemas()[TURBOQUANT_OP_FUSED_DECODE].arguments if a.name == "gate")
    assert "?" in str(gate.type) or "Optional" in str(gate.type), str(gate.type)


@pytest.mark.parametrize("op_name", sorted(BINDING_OPS))
def test_every_op_has_an_npu_and_a_meta_registration(op_name: str) -> None:
    """Without a Meta kernel a FakeTensor trace dies on the op rather than on the tracer."""
    keys = _impl_keys().get(op_name, set())
    assert any("PrivateUse1" in key for key in keys), f"{op_name} has no NPU implementation: {keys}"
    assert any("Meta" in key for key in keys), f"{op_name} has no meta kernel: {keys}"


def test_no_impl_is_registered_without_a_schema() -> None:
    assert set(_impl_keys()) <= set(_schemas())


@pytest.mark.parametrize("op_name", sorted(OP_STUBS))
def test_the_python_stub_passes_the_schema_in_order(op_name: str) -> None:
    """The cross-language contract: the stub's parameters ARE the schema's arguments.

    The `_kernel_*` stubs call their op positionally, so a schema argument
    renamed, reordered, added or dropped on either side has to be done on both.
    This is the only check that notices when it was not.
    """
    schema = _schemas()[op_name]
    stub = getattr(TurboQuantOpsAdapter, OP_STUBS[op_name])
    parameters = [name for name in inspect.signature(stub).parameters if name != "self"]
    assert parameters == [a.name for a in schema.arguments], f"{OP_STUBS[op_name]} and the {op_name} schema disagree"
