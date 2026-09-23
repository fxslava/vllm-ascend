import torch
from torch.library import Library

from vllm_ascend.utils import is_310p

# This file provides a template and registration utilities for writing "meta" implementations
# of custom operators in Python for the vllm_ascend project.
#
# We offer two ways to implement meta implementations for custom ops:
#   1. Python meta implementation (as shown in this file): Write a Python function that
#      takes the same arguments as your operator and returns empty tensors with the correct
#      shapes and dtypes. This is useful for rapid prototyping and for ops that are only
#      used in Python.
#   2. C++ meta implementation: You can also implement the meta function in C++ for better
#      performance or to match the C++ op logic more closely. See `torch_binding_meta.cpp`
#      for examples of C++ meta implementations and how to register them.
#
# Both approaches enable tracing, export, and shape inference in PyTorch and vLLM, which
# is essential for supporting `torch.compile` and aclgraph.

# How to add a new meta implementation in Python:
# -------------------------------------
# 1. Write a Python function that takes the same arguments as your operator, and returns
#    empty tensors (using torch.empty_like, torch.empty, etc.) with the correct shapes and dtypes.
#    Do NOT perform any real computation or allocate device memory.
#
# 2. Register your meta function using `register_meta_if_necessary`, providing:
#    - The namespace (usually "_C_ascend" for custom ops)
#    - The operator name (as registered in C++)
#    - The Python meta function
#    - (Optional) The overload name, if your op has overloads
#
# 3. The registration utility will check if a meta implementation already exists for your op,
#    and only register if necessary. This avoids duplicate registrations.
#
# 4. When developing new custom ops, always provide a meta implementation to enable tracing,
#    export, and shape inference in PyTorch and vLLM to enable the capture of `torch.compile`
#    and aclgraph.
#
# For more details, see: https://pytorch.org/docs/stable/notes/extending.html#meta-tensors

lib = Library("_C_ascend", "IMPL")


def register_meta_if_necessary(ns: str, op_name: str, fn, overload: str = ""):
    if overload != "":
        op_name = op_name + "." + overload
    schema_to_find = ns + "::" + op_name
    meta_impl_list = torch._C._dispatch_get_registrations_for_dispatch_key("Meta")
    if schema_to_find in meta_impl_list:
        return
    lib.impl(op_name, fn, "Meta")


def _turboquant_noop_meta(*args, **kwargs) -> None:
    """The meta kernel every TurboQuant operator needs.

    All five write through ``Tensor!`` arguments the caller already owns -- the
    two cache planes and the scale plane for a writer, the rotated-query buffer,
    the reduction workspace and the attention output for a decode -- and return
    ``()``. So there is no shape to infer: the fake tensors the tracer is holding
    for those buffers are already the right shape and dtype, and a meta kernel
    that allocated anything would be describing an output the operator does not
    have.

    Without one, a TurboQuant operator reached under FakeTensor -- ``torch.export``,
    or a full graph that does not split attention out -- fails with "could not run
    with Meta backend" rather than tracing. The served path does not depend on it:
    both attention entry points (``vllm::unified_attention_with_output`` and
    ``vllm::turboquant_gated_attention``) are graph splitting ops, so under
    piecewise compilation these operators run eagerly and are never traced.

    ``npu_turboquant_vector_core_num`` and the two ``*_workspace_size`` operators
    are deliberately absent: they take no tensors and are registered in C++ without
    a dispatch key, so their catch-all kernel already answers under every backend.
    """
    return None


# Name by name rather than by prefix: a registration is a promise that the operator
# really does only write through its own arguments, which is worth restating per
# operator rather than inferring from a naming convention.
TURBOQUANT_MUTATING_OPS = (
    "npu_turboquant_reshape_and_cache",
    "npu_turboquant_rotate_q",
    "npu_turboquant_paged_attention",
    # Ascend 950 builds only (VLLM_ENABLE_TURBOQUANT_CUBE); absent elsewhere.
    "npu_turboquant_cube_reshape_and_cache",
    "npu_turboquant_cube_decode",
)


def register_turboquant_meta() -> None:
    """Give every TurboQuant operator this build registered a meta kernel.

    Guarded per operator rather than as a block: the kv4fp8 Cube operators exist
    only in an Ascend 950 build, and registering a meta kernel for a schema that
    was never defined raises.
    """
    for op_name in TURBOQUANT_MUTATING_OPS:
        if not hasattr(torch.ops._C_ascend, op_name):
            continue
        register_meta_if_necessary("_C_ascend", op_name, _turboquant_noop_meta)


def bgmv_expand_meta(
    x: torch.Tensor, weight: torch.Tensor, indices: torch.Tensor, y: torch.Tensor, slice_offset: int, slice_size: int
):
    y_out = torch.empty_like(y)
    return y_out


def sgmv_expand_meta(
    x: torch.Tensor,
    weight: torch.Tensor,
    lora_indices: torch.Tensor,
    seq_len: torch.Tensor,
    y: torch.Tensor,
    slice_offset: int,
    slice_size: int,
):
    y_out = torch.empty_like(y)
    return y_out


if not is_310p():
    register_meta_if_necessary("_C_ascend", "bgmv_expand", bgmv_expand_meta)
    register_meta_if_necessary("_C_ascend", "sgmv_expand", sgmv_expand_meta)
    register_turboquant_meta()
