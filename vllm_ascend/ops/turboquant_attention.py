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
"""The attention call of an ``attn_output_gate`` layer whose TurboQuant decode gates in its own launch.

``unified_attention_with_output`` has no argument for the gate, so a layer whose
backend impl reports ``fuses_output_gate`` calls ``vllm::turboquant_gated_attention``
instead and receives ``sigmoid(gate) * attention`` back, ready for ``o_proj``
(see :mod:`vllm_ascend.attention.turboquant_v1`). Like the unified op, it is a
graph splitting op; ``NPUPlatform.check_and_update_config`` registers it as one
under ``vllm_ascend.utils.TURBOQUANT_GATED_ATTENTION_OP``.
"""

import torch
from vllm.utils.torch_utils import direct_register_custom_op


def turboquant_fuses_output_gate(attn: torch.nn.Module) -> bool:
    """Whether ``attn``'s backend applies the output gate itself."""
    return bool(getattr(getattr(attn, "impl", None), "fuses_output_gate", False))


def turboquant_gated_attention_forward(
    attn: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """``Attention.forward`` for a gated layer: returns ``sigmoid(gate) * attention(q, k, v)``, flattened."""
    num_heads = attn.num_heads
    num_kv_heads = attn.num_kv_heads
    head_size = attn.head_size
    output = torch.empty((query.shape[0], num_heads * head_size), dtype=query.dtype, device=query.device)
    torch.ops.vllm.turboquant_gated_attention(
        query.view(-1, num_heads, head_size),
        key.view(-1, num_kv_heads, head_size),
        value.view(-1, num_kv_heads, head_size),
        gate.reshape(-1, num_heads, head_size),
        output.view(-1, num_heads, head_size),
        attn.layer_name,
    )
    return output


def turboquant_gated_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    # Deferred: the attention layer module pulls in most of vLLM's model executor.
    from vllm.model_executor.layers.attention.attention import get_attention_context

    attn_metadata, attn_layer, kv_cache, _ = get_attention_context(layer_name)
    attn_layer.impl.forward(
        attn_layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_gate=gate,
    )


def turboquant_gated_attention_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="turboquant_gated_attention",
    op_func=turboquant_gated_attention,
    mutates_args=["output"],
    fake_impl=turboquant_gated_attention_fake,
    dispatch_key="PrivateUse1",
)
