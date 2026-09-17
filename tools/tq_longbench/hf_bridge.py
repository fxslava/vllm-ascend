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
"""Give a Hugging Face model's full-attention layers the TurboQuant cache, and nothing else.

:mod:`tq_longbench.layers` writes a model out so that nothing sits between a
token and a kernel.  That is the right shape for a dense GQA model and the wrong
shape for a hybrid one: Qwen3.5 is 18 Gated DeltaNet layers to 6 full-attention
layers, and linear attention has no KV cache for TurboQuant to touch.
Reimplementing the other 18 would put three quarters of the model under test
when none of it is what changed.

So this module inverts the arrangement.  ``transformers`` keeps the weights, the
linear attention, the multimodal RoPE and the vision tower; the KV cache and the
attention of the ``full_attention`` layers -- exactly the surface TurboQuant
replaces -- come from here.  That is also how the plugin works in production:
the framework owns the model and the backend owns attention.

The seam is ``ALL_ATTENTION_FUNCTIONS``, which is the supported way to supply an
attention implementation, so no method is monkeypatched and no forward is
copied.  A registered function is handed ``query``, ``key`` and ``value`` after
the projections, the norms, RoPE and the cache update, which is precisely the
contract the TurboQuant operators want: ``Pi`` is applied to what RoPE produced.

What the bridge deliberately does not do is apply the output gate.  Qwen3.5
multiplies by ``sigmoid(gate)`` *after* the attention call, in the layer's own
forward, so the backend is asked for ungated attention and the model gates it --
gating in both places would square the sigmoid.  The Cube decode's fused gate is
a property of the plugin's call path, not of this one.
"""

from __future__ import annotations

import torch

from tq_longbench.kv_cache import CacheGeometry
from tq_longbench.ops import AttentionBackend, LayerShape, build_backend

#: One sequence at a time, as everywhere else in the harness. A padded batch
#: would need the attention mask this bridge ignores in favour of context
#: lengths, so it is refused rather than attending over the padding in silence.
_SUPPORTED_BATCH = 1

#: Names one registration per bridge, monotonically. See :meth:`install`.
_BRIDGE_SERIAL = 0


class TurboQuantAttentionBridge:
    """Routes a model's full-attention layers through a :class:`~tq_longbench.ops.AttentionBackend`.

    Install it with :meth:`install`, run the model as usual, and call
    :meth:`reset` between prompts.  The layers it does not claim -- linear
    attention, vision -- are untouched and keep whatever implementation the
    checkpoint was loaded with.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        backend: str,
        max_seq_len: int,
        block_size: int = 128,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.attentions = discover_full_attention_layers(model)
        if not self.attentions:
            raise ValueError(
                "no full-attention layers found: this model has nothing with a KV cache for TurboQuant to hold"
            )
        probe = next(iter(self.attentions.values()))
        self.device = device or next(probe.q_proj.parameters()).device
        self.dtype = dtype or next(probe.q_proj.parameters()).dtype

        head_size = probe.head_dim
        num_kv_heads = probe.k_proj.out_features // head_size
        num_heads = probe.o_proj.in_features // head_size

        #: HF layer index -> the plane this bridge gave it. The full-attention
        #: layers are sparse in a hybrid model (3, 7, 11, ... in Qwen3.5), so the
        #: cache holds one plane per *claimed* layer rather than per model layer.
        self.plane_of = {index: plane for plane, index in enumerate(sorted(self.attentions))}

        self.geometry = CacheGeometry(
            num_layers=len(self.attentions),
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_size=block_size,
            max_seq_len=max_seq_len,
        )
        self.backend: AttentionBackend = build_backend(
            backend,
            self.geometry,
            LayerShape(num_heads, num_kv_heads, head_size, probe.scaling),
            self.device,
            self.dtype,
        )
        # A counter, not id(self): ids are reused once an object is collected,
        # and two bridges sharing a name would silently share a registration.
        global _BRIDGE_SERIAL
        _BRIDGE_SERIAL += 1
        self._name = f"tq_longbench_{_BRIDGE_SERIAL}"
        self._installed: list[tuple[object, str]] = []

    # ------------------------------------------------------------- lifecycle

    def install(self) -> TurboQuantAttentionBridge:
        """Register the implementation and point the claimed layers at it.

        The layers of one model **share a config object**, so the
        implementation each was on is recorded once per distinct config and
        before anything is written to it. Recording it per layer instead reads
        back this bridge's own name from the second layer onward, and
        :meth:`uninstall` then restores the model onto the bridge it was meant
        to be leaving -- which does not raise, it just silently keeps
        intercepting, and the next measurement compares the backend against
        itself.
        """
        from transformers import AttentionInterface

        if self._installed:
            raise RuntimeError("this bridge is already installed; uninstall it before installing again")

        AttentionInterface.register(self._name, self._attention)
        original: dict[int, tuple[object, str]] = {}
        for module in self.attentions.values():
            config = module.config
            original.setdefault(id(config), (config, config._attn_implementation))
        for config, _ in original.values():
            config._attn_implementation = self._name
        self._installed = list(original.values())
        return self

    def uninstall(self) -> None:
        """Put every config back on the implementation it had before :meth:`install`."""
        for config, previous in self._installed:
            config._attn_implementation = previous
        self._installed.clear()

    def __enter__(self) -> TurboQuantAttentionBridge:
        return self.install()

    def __exit__(self, *exc_info) -> None:
        self.uninstall()

    def reset(self) -> None:
        """Zero every plane, so the next prompt cannot read the last one's tokens.

        The pools are static and the slots are positions, so without this a
        shorter second prompt would leave the first one's tail in place --
        reachable only through a context length that is too long, which the tie
        point catches, but cheap enough to make impossible instead.
        """
        for plane in (*getattr(self.backend.cache, "key_planes", ()), *getattr(self.backend.cache, "value_planes", ())):
            plane.zero_()
        for plane in getattr(self.backend.cache, "scale_planes", ()):
            plane.zero_()

    def memory_report(self) -> dict:
        dense = 2 * self.geometry.num_layers * self.geometry.num_blocks * self.geometry.block_size
        dense *= self.geometry.num_kv_heads * self.geometry.head_size * torch.empty((), dtype=self.dtype).element_size()
        held = self.backend.cache.bytes_allocated()
        return {
            "claimed_layers": len(self.attentions),
            "kv_cache_mb": held / (1024 * 1024),
            "dense_equivalent_mb": dense / (1024 * 1024),
            "bytes_per_token_per_layer": self.backend.cache.bytes_per_token_per_layer(),
        }

    # -------------------------------------------------------------- the hook

    def _attention(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        scaling: float | None = None,
        dropout: float = 0.0,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        """``q``, ``k``, ``v`` are ``[batch, heads, seq, head_size]``; the answer is ``[batch, seq, heads, head_size]``.

        ``key`` and ``value`` arrive holding the *whole* prefix, because the
        layer has already folded the new tokens into its own cache. The new ones
        are therefore its tail, and the prefix length is the whole of it -- so
        the position a token is written to needs no counter kept on the side,
        which is one fewer thing that can drift out of step with the cache.
        """
        batch, _, new_tokens, head_size = query.shape
        if batch != _SUPPORTED_BATCH:
            raise NotImplementedError(
                f"the TurboQuant bridge runs one sequence at a time, got a batch of {batch}. A padded batch "
                "needs the attention mask this bridge replaces with per-token context lengths."
            )
        prefix_end = key.shape[2]
        start = prefix_end - new_tokens
        plane = self.plane_of[module.layer_idx]

        # [B, H, T, D] -> [T, H, D]; the operators are token-major.
        queries = query[0].transpose(0, 1).contiguous()
        new_keys = key[0, :, start:, :].transpose(0, 1).contiguous()
        new_values = value[0, :, start:, :].transpose(0, 1).contiguous()

        self.backend.write_kv(plane, new_keys, new_values, self.backend.cache.slot_mapping(start, new_tokens))

        out = torch.empty_like(queries)
        if new_tokens == 1:
            lengths = torch.full((1,), prefix_end, dtype=torch.int32, device=query.device)
            self.backend.check_tie_point(lengths, prefix_end)
            self.backend.decode(plane, queries, lengths, out)
        else:
            self.backend.prefill_chunk(plane, queries, prefix_end, out)

        # The layer applies sigmoid(gate) to what it gets back, so this must be
        # ungated. See the module docstring.
        return out.unsqueeze(0), None


def discover_full_attention_layers(model: torch.nn.Module) -> dict[int, torch.nn.Module]:
    """Return ``{layer_idx: attention module}`` for every layer with a real KV cache.

    Structural rather than by class name, so the same bridge serves a hybrid
    model and a plain dense one: a full-attention layer is one that projects
    q, k, v and o and carries a ``layer_idx``.  Gated DeltaNet has ``in_proj_qkv``
    and no ``o_proj``, so it is not claimed -- which is correct, because linear
    attention keeps a recurrent state rather than a KV cache.
    """
    found: dict[int, torch.nn.Module] = {}
    for module in model.modules():
        if not all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj", "layer_idx")):
            continue
        if not hasattr(module, "head_dim") or not hasattr(module, "scaling"):
            continue
        index = module.layer_idx
        if index is None:
            continue
        if index in found:
            raise ValueError(
                f"two attention modules report layer_idx {index}; the bridge cannot tell their caches apart"
            )
        found[index] = module
    return found
