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
"""The TurboQuant cache in plain PyTorch, so the harness runs where the kernels do not.

The Ascend C operators exist on one vendor's hardware.  Everything above them --
the paging, the tie point, the chunked prefill, the greedy loop, and the
question of whether a 4-bit cache still answers a long-context prompt -- is
hardware-independent, and waiting for an NPU to ask it is a poor trade.  These
backends implement the same arithmetic in device-agnostic torch, so a run on
CUDA or on a CPU exercises the harness end to end and quantifies the
quantisation loss on real weights.

They are a **reference, not a simulation**: the arithmetic matches (the packing,
the Lloyd-Max bins, the RMS scale, the rotation, the scale plane's burst
padding), the performance emphatically does not, and nothing here says anything
about what a kernel costs.  For latency, run the real operators.

The correctness argument is transitive.  ``tests/ut/attention/turboquant_cpu_ops.py``
is already held to the kernels' own goldens; this module is held to *it*, tensor
for tensor, by ``TestReferenceMatchesTheStandIns``.  So a number produced here on
CUDA is the number the kernel produces, up to float ordering.

Two backends:

* :class:`DenseReferenceBackend` -- the unquantised baseline, over the same
  paged pool, through ``scaled_dot_product_attention``.  Stands in for
  ``npu_fused_infer_attention_score``.
* :class:`TurboQuantReferenceBackend` -- the 4-bit path: rotate, quantise, pack,
  and on the way back dequantise, attend, un-rotate and gate.
"""

from __future__ import annotations

import torch

from tq_longbench._ascend import turboquant_layout, turboquant_rotation
from tq_longbench.kv_cache import CacheGeometry, DenseKVCache, TurboQuantKVCache
from tq_longbench.ops import AttentionBackend, LayerShape, longest_context

# TurboQuantCodec<4>: two 4-bit codes per byte, stored biased into int8.
_NIBBLE_RADIX = 16
_INT8_BIAS = 128

# The scale floor that keeps an all-zero vector from dividing by zero;
# TurboQuantCodec<4>::kEps.
_EPS = 1e-20

#: Context positions dequantised at once. The score matrix is
#: ``tokens x heads x window``, and at 128k with head_size 256 a single pass
#: would be hundreds of megabytes of fp32 for no gain -- the softmax is
#: accumulated across windows instead.
_CONTEXT_WINDOW = 8192

#: fp32 elements the score matrix may hold, ~256 MB. A prefill chunk attends
#: with thousands of query tokens at once, where a fixed window would scale the
#: matrix by the chunk size and run a 12 GB card out of memory; the window is
#: divided down to fit instead. Purely a memory bound: the streaming softmax
#: makes the answer independent of it, which
#: ``test_the_streaming_softmax_does_not_depend_on_the_window`` holds it to.
_SCORE_BUDGET = 64 * 1024 * 1024

_MIN_CONTEXT_WINDOW = 256


def context_window(tokens: int, heads: int) -> int:
    """How much context to dequantise at once, given how many rows are attending."""
    affordable = _SCORE_BUDGET // max(1, tokens * heads)
    return max(_MIN_CONTEXT_WINDOW, min(_CONTEXT_WINDOW, affordable))


def quantize(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``Quantize4Bit`` over the last axis: RMS scale, strict Lloyd-Max bins, low nibble first."""
    head_size = vectors.shape[-1]
    scale = vectors.pow(2).sum(dim=-1, keepdim=True).sqrt() / head_size**0.5 + _EPS
    thresholds = torch.tensor(
        turboquant_layout().TURBOQUANT_LLOYD_MAX_THRESHOLDS, dtype=vectors.dtype, device=vectors.device
    )
    bins = ((vectors / scale).unsqueeze(-1) > thresholds).sum(dim=-1)
    packed = bins[..., 0::2] + _NIBBLE_RADIX * bins[..., 1::2] - _INT8_BIAS
    return packed.to(torch.int8), scale.squeeze(-1)


def dequantize(packed: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """``Dequantize4Bit``: one bare level per channel, before the per-vector scale."""
    byte = packed.to(torch.int64) + _INT8_BIAS
    codes = torch.stack((byte % _NIBBLE_RADIX, byte // _NIBBLE_RADIX), dim=-1).flatten(-2)
    return centroids[codes]


def centroid_table(device: torch.device) -> torch.Tensor:
    """The Lloyd-Max reconstruction levels, where the kernel reads them from the codec image."""
    return torch.tensor(turboquant_layout().TURBOQUANT_LLOYD_MAX_CENTROIDS, dtype=torch.float32, device=device)


class DenseReferenceBackend(AttentionBackend):
    """The unquantised baseline in torch: the same paged pool, through SDPA.

    Present so that ``native_v5``'s role -- a number that is not a quantisation
    of anything -- survives onto a host with no Ascend runtime.  The paging is
    real; only the attention call is torch's.
    """

    name = "dense_reference"
    # SDPA over a mask this builds from the lengths on the device, and a
    # scatter of fixed shape to write: with the window fixed, a step asks the
    # host nothing.
    supports_static_decode = True
    supports_graph_capture = True

    def __init__(
        self,
        geometry: CacheGeometry,
        shape: LayerShape,
        device: torch.device,
        dtype: torch.dtype,
        output_rotation_folded: bool = False,
    ) -> None:
        super().__init__(geometry, shape, device)
        self.cache = DenseKVCache(geometry, device, dtype)
        self.dtype = dtype
        self.output_rotation_folded = output_rotation_folded

    def write_kv(self, layer: int, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor) -> None:
        self.cache.write(layer, key, value, slots)

    def _gathered(self, layer: int, context: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(k, v)`` for the first ``context`` positions, as ``[heads, context, head_size]``.

        Left in the cache's own dtype and *not* expanded over the query group:
        ``scaled_dot_product_attention(..., enable_gqa=True)`` broadcasts the kv
        heads itself, and materialising the expansion at 32k context is hundreds
        of megabytes that exist only to be read once.
        """
        packed = (-1, self.geometry.num_kv_heads, self.geometry.head_size)
        rows = slice(0, context)
        return (
            self.cache.key_planes[layer].view(packed)[rows].transpose(0, 1),
            self.cache.value_planes[layer].view(packed)[rows].transpose(0, 1),
        )

    def _attend(self, layer: int, query: torch.Tensor, mask: torch.Tensor, context: int) -> torch.Tensor:
        keys, values = self._gathered(layer, context)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1),
            keys,
            values,
            attn_mask=mask,
            scale=self.shape.scale,
            enable_gqa=self.shape.num_heads != self.shape.num_kv_heads,
        )
        return attended.transpose(0, 1)

    def decode(
        self,
        layer: int,
        query: torch.Tensor,
        context_lens: torch.Tensor,
        out: torch.Tensor,
        gate: torch.Tensor | None = None,
        window: int | None = None,
    ) -> torch.Tensor:
        longest = window if window is not None else longest_context(context_lens)
        positions = torch.arange(longest, device=query.device).view(1, -1)
        # One row per query token, each seeing exactly its own prefix.
        mask = (positions < context_lens.to(query.device).view(-1, 1)).unsqueeze(0)
        out.copy_(self._attend(layer, query.contiguous(), mask, longest).to(out.dtype))
        if gate is not None:
            out.mul_(torch.sigmoid(gate).view_as(out))
        return self._into_folded_basis(out)

    def prefill_chunk(
        self,
        layer: int,
        query: torch.Tensor,
        prefix_end: int,
        out: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Bottom-right causal over the whole prefix, in one attention call.

        The base class spells a chunk out as one request per token, which is
        correct and costs a ``chunk x prefix`` score matrix. Handing SDPA the
        mask instead lets it pick a kernel that never materialises one, which is
        what makes a 32k prefill fit on a 12 GB card.
        """
        count = query.shape[0]
        first = prefix_end - count
        if first < 0:
            raise ValueError(f"a {count}-token chunk cannot end at prefix {prefix_end}")
        rows = torch.arange(first, prefix_end, device=query.device).view(-1, 1)
        columns = torch.arange(prefix_end, device=query.device).view(1, -1)
        mask = (columns <= rows).unsqueeze(0)
        out.copy_(self._attend(layer, query.contiguous(), mask, prefix_end).to(out.dtype))
        if gate is not None:
            out.mul_(torch.sigmoid(gate).view_as(out))
        return self._into_folded_basis(out)


class TurboQuantReferenceBackend(AttentionBackend):
    """The 4-bit cache in torch: rotate, quantise, pack -- then dequantise, attend, un-rotate.

    Mirrors ``npu_turboquant_reshape_and_cache`` and
    ``npu_turboquant_paged_attention`` rather than the Cube pair, because the
    vector path is the one with a reference to be held to.  The output stage is
    the same decision the Cube kernel's epilogue makes: leave it rotated for a
    folded ``o_proj``, otherwise un-rotate, and multiply by ``sigmoid(gate)``
    when the layer has one.
    """

    name = "turboquant_reference"
    supports_static_decode = True
    # ...but not captured. :meth:`write_kv` drops padding slots with a boolean
    # mask, whose result has as many rows as there were live slots -- a shape
    # that depends on values, which is a host read wherever it appears. The
    # kernels this stands in for have no such step: they hand the slots to the
    # operator and let it skip the padding. Nothing is lost by not capturing a
    # backend whose whole purpose is to compute the right numbers off an NPU.
    supports_graph_capture = False

    def __init__(
        self,
        geometry: CacheGeometry,
        shape: LayerShape,
        device: torch.device,
        output_rotation_folded: bool = False,
    ) -> None:
        super().__init__(geometry, shape, device)
        self.cache = TurboQuantKVCache(geometry, device)
        self.output_rotation_folded = output_rotation_folded
        rotation = turboquant_rotation()
        self._apply_pi = rotation.apply_pi
        self._pi_signs = rotation.turboquant_pi_signs(shape.head_size, device)
        self._centroids = centroid_table(device)

    @property
    def fuses_output_gate(self) -> bool:
        """The un-rotation and the gate happen in the same pass, as they do in the kernel epilogue."""
        return not self.output_rotation_folded

    def write_kv(self, layer: int, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor) -> None:
        heads, head_size = self.shape.num_kv_heads, self.shape.head_size
        key_plane, value_plane, scale_plane = self.cache.planes(layer)
        rows = slots.to(torch.int64)
        written = rows >= 0
        rows = rows[written]
        if rows.numel() == 0:
            return

        # K and V are quantised together so one pass produces both planes'
        # scales, which is also how the kernel fills a token's scale slot.
        planes = torch.cat((key, value), dim=1)[written].to(torch.float32)
        packed, scales = quantize(self._apply_pi(planes, self._pi_signs))

        shape = (-1, heads, head_size // turboquant_layout().TURBOQUANT_PACK_FACTOR)
        key_plane.view(shape)[rows] = packed[:, :heads]
        value_plane.view(shape)[rows] = packed[:, heads:]
        slot = self.geometry.scale_slot
        token_scales = torch.zeros(rows.numel(), slot, dtype=torch.float32, device=self.device)
        token_scales[:, : 2 * heads] = scales
        scale_plane.view(-1, slot)[rows] = token_scales

    def _dequantized(self, layer: int, start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
        """``(k, v)`` for context positions ``[start, stop)``, scaled and head-expanded."""
        heads, head_size = self.shape.num_kv_heads, self.shape.head_size
        key_plane, value_plane, scale_plane = self.cache.planes(layer)
        shape = (-1, heads, head_size // turboquant_layout().TURBOQUANT_PACK_FACTOR)
        rows = slice(start, stop)
        scales = scale_plane.view(-1, self.geometry.scale_slot)[rows]
        keys = dequantize(key_plane.view(shape)[rows], self._centroids) * scales[:, :heads].unsqueeze(-1)
        values = dequantize(value_plane.view(shape)[rows], self._centroids) * scales[:, heads : 2 * heads].unsqueeze(-1)
        group = self.shape.num_heads // heads
        return keys.repeat_interleave(group, dim=1), values.repeat_interleave(group, dim=1)

    def decode(
        self,
        layer: int,
        query: torch.Tensor,
        context_lens: torch.Tensor,
        out: torch.Tensor,
        gate: torch.Tensor | None = None,
        window: int | None = None,
    ) -> torch.Tensor:
        if self.output_rotation_folded and gate is not None:
            raise ValueError(
                "a layer with Pi folded into o_proj cannot carry an output gate: the gate sits between "
                "attention and o_proj, where the output is still rotated"
            )
        longest = window if window is not None else longest_context(context_lens)
        tokens, heads, head_size = query.shape
        device = query.device
        lengths = context_lens.to(device).view(-1, 1, 1)
        # Scores are basis-invariant under Pi, but the cache holds Pi k, so the
        # query has to enter the same basis to meet it.
        rotated_query = self._apply_pi(query.to(torch.float32), self._pi_signs)

        # Streaming softmax over context windows: one running max, one running
        # denominator and one running numerator, so a 128k context never
        # materialises a 128k score matrix.
        running_max = torch.full((tokens, heads, 1), float("-inf"), dtype=torch.float32, device=device)
        denominator = torch.zeros((tokens, heads, 1), dtype=torch.float32, device=device)
        numerator = torch.zeros((tokens, heads, head_size), dtype=torch.float32, device=device)

        window = context_window(tokens, heads)
        for start in range(0, longest, window):
            stop = min(start + window, longest)
            keys, values = self._dequantized(layer, start, stop)
            scores = torch.einsum("thd,lhd->thl", rotated_query, keys) * self.shape.scale
            positions = torch.arange(start, stop, device=device).view(1, 1, -1)
            scores = scores.masked_fill(positions >= lengths, float("-inf"))

            window_max = scores.max(dim=-1, keepdim=True).values
            updated_max = torch.maximum(running_max, window_max)
            # -inf - -inf is NaN; a window entirely masked out contributes nothing.
            updated_max = torch.where(torch.isinf(updated_max), torch.zeros_like(updated_max), updated_max)
            rescale = torch.exp(running_max - updated_max)
            rescale = torch.where(torch.isinf(running_max), torch.zeros_like(rescale), rescale)

            weights = torch.exp(scores - updated_max)
            denominator = denominator * rescale + weights.sum(dim=-1, keepdim=True)
            numerator = numerator * rescale + torch.einsum("thl,lhd->thd", weights, values)
            running_max = updated_max

        attended = numerator / denominator.clamp_min(_EPS)
        if not self.output_rotation_folded:
            attended = self._apply_pi(attended, self._pi_signs)
        if gate is not None:
            attended = attended * torch.sigmoid(gate.to(torch.float32)).view_as(attended)
        out.copy_(attended.to(out.dtype))
        return out


REFERENCE_BACKENDS = {
    DenseReferenceBackend.name: DenseReferenceBackend,
    TurboQuantReferenceBackend.name: TurboQuantReferenceBackend,
}
