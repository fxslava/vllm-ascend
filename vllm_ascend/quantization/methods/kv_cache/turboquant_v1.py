#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""TurboQuant 4-bit KV cache quantization scheme.

The scheme owns nothing but activation: there are no quantization parameters to
load and nothing is rewritten at load time.  The rotation ``Pi = D H D`` is
applied by the Ascend C kernels to K, V and Q at runtime, so ``q_proj``,
``k_proj`` and ``v_proj`` keep the values the checkpoint shipped and RoPE keeps
operating on the unrotated basis.

``o_proj`` is the one exception, and only for a checkpoint folded OFFLINE by
``scripts/tq_fold_output_rotation.py``: the decode output stays rotated and the
folded projection un-rotates it.  The backend reads the fold record from the
model config per layer; see ``vllm_ascend.attention.turboquant_rotation``.

The attention implementation lives in
``vllm_ascend.attention.turboquant_v1``; this module is the registry entry that
points a layer at it.
"""

import torch
from vllm.logger import logger

from vllm_ascend.attention.turboquant_v1 import (
    AscendTurboQuantAttentionBackend,
    activate_turboquant_backend,
)

from ..base import AscendAttentionScheme
from ..registry import register_scheme


@register_scheme("TurboQuant", "attention")
class AscendTurboQuantKVCacheAttentionMethod(AscendAttentionScheme):
    """4-bit rotated KV cache for dense-attention models."""

    # Re-exported so a model runner can ask for the cache geometry without
    # importing the attention module directly.
    backend = AscendTurboQuantAttentionBackend

    def create_weights(self, layer: torch.nn.Module) -> None:
        # The packed cache is int8; two 4-bit codes share a byte and the
        # per-vector scales live in a side allocation owned by the impl.
        layer.kv_cache_torch_dtype = torch.int8
        activate_turboquant_backend(layer)
        logger.info_once(
            "[vllm-ascend/turboquant] 4-bit rotated KV cache enabled; q/k/v projections are left untouched, "
            "and o_proj is used as the checkpoint ships it (Pi-folded or not)"
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Deliberately empty. Folding Pi into q/k/v would put RoPE in the rotated
        # basis, so K, V and Q rotate at runtime. Folding it into o_proj alone is
        # valid, but happens offline, before quantisation, and only for a layer
        # with no output gate -- not here, where o_proj may already be quantised
        # and this layer cannot see it.
        return

    def apply(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache,
        attn_metadata,
        attn_type,
        scale,
        output,
    ) -> torch.Tensor:
        raise RuntimeError(
            "[vllm-ascend/turboquant] AscendTurboQuantKVCacheAttentionMethod.apply should not be called. "
            "TurboQuant KV cache quantization is handled by the attention backend."
        )
