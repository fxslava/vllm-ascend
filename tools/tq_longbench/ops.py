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
"""The four attention paths a run can take, and the buffers they reuse.

Each backend answers the same two questions -- write this chunk's K/V into the
cache, and attend over the prefix -- so the decoder layer never learns which one
it is driving.  What differs is underneath:

* :class:`NativeV5Backend` -- torch_npu's fused infer attention over the dense
  paged cache: ``npu_fused_infer_attention_score_v2`` (aclnn V5, the only
  version Ascend 950 accepts) where torch_npu has it.  The baseline, and the
  only path whose numbers are not a quantisation of anything.
* :class:`CANNDenseBackend` -- the same dense pool with no fused kernel over it
  at all: matmul, softmax, matmul, each a CANN operator torch_npu launches.
  Slower, and the unquantised baseline that survives where op-plugin's FIA does
  not -- which on Ascend 950 is everywhere (``EZ9903``).
* :class:`TurboQuantCubeBackend` -- one ``npu_turboquant_cube_decode`` launch
  per step.  It takes the **raw** query (the ``PRE_ROTATED = false`` entry),
  rotates it in its own prologue, and its output stage un-rotates the result and
  applies ``sigmoid(gate)`` before the fp16 write.  kv4fp8, float16 only.
* :class:`TurboQuantAivBackend` -- ``npu_turboquant_rotate_q`` then
  ``npu_turboquant_paged_attention``, then the rotation again over the output
  unless the layer's ``o_proj`` is folded.  Every other build and dtype, and the
  only TurboQuant path with a CPU stand-in, so it is what the harness's own
  tests hold to exact attention.

The scratch buffers live on the backend rather than inside a launch for the
reason they do in the plugin: an allocation on the decode step is an allocation
the step's latency percentiles measure, and an address a captured graph cannot
replay.  They are grown here, never in :meth:`decode`.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import torch

from tq_longbench._ascend import load_turboquant_library, turboquant_layout, turboquant_rotation
from tq_longbench.kv_cache import CacheGeometry, DenseKVCache, TurboQuantKVCache

# The codec table image is built for a batch of rows; the writer and the query
# rotation use one, the paged decode tiles 16 at a time.
_ROTATE_BATCH_ROWS = 1

_ASCEND_NAMESPACE = "_C_ascend"

# FIA's (and PFA's) bottom-right causal mask: with more kv than q the diagonal
# aligns to the end of the prefix, which is what a chunk arriving after a
# populated cache needs.
_BOTTOM_RIGHT_CAUSAL_SPARSE_MODE = 3


def ascend_ops():
    """The operator namespace the extension registers into.

    When nothing has registered the operators, a standalone library
    (``tools/tq_longbench/build_turboquant_ops.py``) is loaded if one is found;
    the check is free once they are there. Raised on rather than returned as
    ``None``: every caller needs it, and an ``AttributeError`` from deep inside
    a launch says far less than this does.
    """
    if not _operators_served():
        outcome = load_turboquant_library(_operators_served)
        if not outcome.registered:
            raise RuntimeError(
                "the TurboQuant operators are not registered. Build vllm-ascend's C++ extension with "
                "VLLM_ENABLE_TURBOQUANT (and VLLM_ENABLE_TURBOQUANT_CUBE for the Cube decode), build the standalone "
                "library with tools/tq_longbench/build_turboquant_ops.py, or run under "
                "tests.ut.attention.turboquant_cpu_ops.turboquant_cpu_ops() to serve them from the CPU. "
                f"Standalone library: {outcome.detail}."
            )
    return getattr(torch.ops, _ASCEND_NAMESPACE)


def cube_decode_available() -> bool:
    """Whether this build registered the kv4fp8 Cube kernels (Ascend 950 builds only)."""
    ops = getattr(torch.ops, _ASCEND_NAMESPACE, None)
    return ops is not None and hasattr(ops, "npu_turboquant_cube_decode")


@dataclass(frozen=True)
class LayerShape:
    """What a backend needs to know about the layer it is serving."""

    num_heads: int
    num_kv_heads: int
    head_size: int
    scale: float

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads:
            raise ValueError(f"num_heads {self.num_heads} is not a multiple of num_kv_heads {self.num_kv_heads}")


def longest_context(context_lens: torch.Tensor) -> int:
    """The longest context length, read back to the host.

    Named rather than inlined because it is the synchronisation the static decode
    path exists to remove: every call is a device-to-host copy, and on the decode
    it happens once per layer per step. A backend handed a ``window`` never
    reaches here.
    """
    return int(context_lens.max()) if context_lens.numel() else 0


class AttentionBackend(abc.ABC):
    """One decode path, bound to one cache pool, serving every layer of the model."""

    name: str
    #: Whether :meth:`decode` applies ``sigmoid(gate)`` itself. When False the
    #: caller multiplies in torch, so a gated layer is correct either way.
    fuses_output_gate: bool = False
    #: Whether the ``o_proj`` this backend feeds has Pi folded into it, so the
    #: attention output must leave in the rotated basis, ``O Pi``.
    output_rotation_folded: bool = False
    #: Whether :meth:`decode` runs with no device-to-host copy when it is given a
    #: ``window``. False means every step reads its own context lengths back --
    #: correct, but a stall per layer per step, and not something a graph can be
    #: captured around: the lengths would be frozen at the length the capture saw.
    supports_static_decode: bool = False
    #: Whether *every* call this backend makes on the decode path -- the cache
    #: write as well as the decode -- has a shape fixed before it runs. This is
    #: the stricter of the two and the one graph capture needs, because a shape
    #: that depends on a tensor's values is a host read wherever it appears.
    #: Separate from :attr:`supports_static_decode` because the two can differ:
    #: :class:`~tq_longbench.reference.TurboQuantReferenceBackend` decodes
    #: without asking the host anything and still cannot be captured, because its
    #: write filters padding slots with a boolean mask.
    supports_graph_capture: bool = False

    def __init__(self, geometry: CacheGeometry, shape: LayerShape, device: torch.device) -> None:
        self.geometry = geometry
        self.shape = shape
        self.device = device
        #: ``(num_tokens, window) -> block table``; see :meth:`_block_tables`.
        self._expanded_block_tables: dict[tuple[int, int], torch.Tensor] = {}

    def _into_folded_basis(self, out: torch.Tensor) -> torch.Tensor:
        """Rotate an unquantised attention output by Pi when the projection expects that.

        A dense backend attends in the model's own basis, so behind a folded
        ``o_proj`` its output must be rotated before the projection un-rotates
        it. This is ``dense_staging``'s prefill whenever the decode is folded:
        skipping it would feed every prefill token's attention through the wrong
        basis, and nothing downstream would raise.
        """
        if self.output_rotation_folded:
            rotation = turboquant_rotation()
            signs = rotation.turboquant_pi_signs(self.shape.head_size, out.device)
            out.copy_(rotation.apply_pi(out.to(torch.float32), signs).to(out.dtype))
        return out

    @abc.abstractmethod
    def write_kv(self, layer: int, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor) -> None:
        """Write ``[num_tokens, num_kv_heads, head_size]`` K/V at the named slots."""

    @abc.abstractmethod
    def decode(
        self,
        layer: int,
        query: torch.Tensor,
        context_lens: torch.Tensor,
        out: torch.Tensor,
        gate: torch.Tensor | None = None,
        window: int | None = None,
    ) -> torch.Tensor:
        """Attend ``query`` ``[num_tokens, num_heads, head_size]`` over each token's prefix.

        ``context_lens`` holds one length per query token, so the same call
        serves a single decode step and a whole prefill chunk presented as a
        batch of per-position decodes.

        ``window`` fixes how many cache slots the call reads. Without it each
        backend asks ``context_lens`` for its longest entry, which is a
        device-to-host copy -- one per layer per step, and illegal inside a graph
        capture. With it the shape is a constant the caller chose and the lengths
        are only ever read on the device, as the mask they already were.
        It must cover every length in ``context_lens``; slots past a row's own
        length are masked off exactly as they always were, so a window wider
        than the prefix costs reads and changes no number.
        """

    def prefill_chunk(
        self,
        layer: int,
        query: torch.Tensor,
        prefix_end: int,
        out: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Attend a whole chunk whose last token sits at absolute position ``prefix_end - 1``.

        The default presents the chunk as a batch of per-position decodes: query
        token ``i`` gets context length ``prefix_end - count + i + 1``, which is
        bottom-right causal spelled out one row at a time.  It is what the
        TurboQuant kernels can do -- there is no paged prefill over the 4-bit
        cache -- and it is exact, just billed per token.
        """
        count = query.shape[0]
        first = prefix_end - count
        if first < 0:
            raise ValueError(f"a {count}-token chunk cannot end at prefix {prefix_end}")
        context_lens = torch.arange(first + 1, prefix_end + 1, dtype=torch.int32, device=query.device)
        return self.decode(layer, query, context_lens, out, gate, window=prefix_end)

    def _block_tables(self, num_tokens: int, window: int) -> torch.Tensor:
        """``[num_tokens, blocks]`` for this window, built once per shape it is asked for.

        ``block_table(window).expand(n, -1).contiguous()`` is a fresh allocation
        and a copy every time it is called -- once per layer per step, for a
        tensor whose contents depend on nothing but the two numbers here. Cached,
        it is neither, and under graph capture it is an address the replay
        already holds rather than a pointer into the graph's pool.

        Keyed on the window rather than on the context length, so a captured
        graph and the steps that follow it share one entry.
        """
        key = (num_tokens, window)
        table = self._expanded_block_tables.get(key)
        if table is None:
            table = self.cache.block_table(window).expand(num_tokens, -1).contiguous()
            self._expanded_block_tables[key] = table
        return table

    def check_tie_point(self, context_lens: torch.Tensor, expected_prefix: int) -> None:
        """Refuse a decode whose kv length disagrees with the prefix actually written.

        The failure this catches is the quiet one: a decode told the context is
        shorter than it is attends to a truncated prefix and returns a fluent,
        wrong answer, and a decode told it is longer reads slots that were never
        written -- zeros, which decode to a centroid rather than to nothing.
        Neither raises anywhere else.
        """
        longest = int(context_lens.max()) if context_lens.numel() else 0
        if longest != expected_prefix:
            raise RuntimeError(
                f"actualSeqLengthsKv tie-point check failed: the longest context length handed to the decode is "
                f"{longest} but {expected_prefix} tokens have been written to the cache. The decode would attend "
                "over a prefix that is not the one prefill built."
            )
        if longest > self.geometry.max_seq_len:
            raise RuntimeError(f"context length {longest} exceeds the {self.geometry.max_seq_len}-token static cache")


#: The torch_npu entry points :class:`NativeV5Backend` can attend through, most preferred first.
#: ``fia_v5`` is ``npu_fused_infer_attention_score_v2``, which drives aclnnFusedInferAttentionScoreV5 --
#: the only FIA interface Ascend 950 accepts ("versions V1 to V4 are no longer supported", EZ9903)
#: and the one vllm-ascend's own attention calls. ``fia_v1`` is the original entry point (V1-V4),
#: still the one older torch_npu builds have. ``pfa`` is ``npu_prompt_flash_attention`` over the
#: contiguous prefix -- it has no block table, so it serves prefill only.
DENSE_ATTENTION_APIS = {
    "fia_v5": "npu_fused_infer_attention_score_v2",
    "fia_v1": "npu_fused_infer_attention_score",
    "pfa": "npu_prompt_flash_attention",
}

#: The APIs that can decode: a batch of single-token rows over a paged prefix needs a block table.
_DECODE_CAPABLE_APIS = ("fia_v5", "fia_v1")

#: sparse_mode 2/3/4 read a compressed 2048 x 2048 causal mask rather than one per shape --
#: the int8 upper triangle vllm-ascend builds (AttentionMaskBuilder.get_splitfuse_attn_mask).
_COMPRESSED_CAUSAL_MASK_SIZE = 2048

#: ``next_tokens`` for a causal call, as vllm-ascend passes it next to sparse_mode 3.
_CAUSAL_NEXT_TOKENS = 0

#: float32 scores one :class:`CANNDenseBackend` tile may hold: 64Mi elements. That is
#: 256 MB of scores, 256 MB again for the softmax's output and 64 MB for the mask --
#: ~576 MB of HBM live at the peak, chosen to leave a 64 GB device most of itself for
#: the weights and the pools.
_SCORE_TILE_ELEMENTS = 64 * 1024 * 1024


def _torch_npu():
    """Deferred: only the native backend needs the device runtime."""
    import torch_npu

    return torch_npu


def available_dense_attention_apis(role: str = "prefill") -> list[str]:
    """The :data:`DENSE_ATTENTION_APIS` this torch_npu has, most preferred first.

    ``role`` is ``"prefill"`` or ``"decode"``; PFA is offered for prefill only.
    Having an entry point says nothing about whether this SoC's CANN accepts it --
    which is what :mod:`tq_longbench.preflight` finds out.
    """
    torch_npu = _torch_npu()
    candidates = DENSE_ATTENTION_APIS if role == "prefill" else _DECODE_CAPABLE_APIS
    return [api for api in candidates if hasattr(torch_npu, DENSE_ATTENTION_APIS[api])]


class _DensePagedBackend(AttentionBackend):
    """Shared plumbing for the two unquantised paths: the fp16 pool and the output stage."""

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

    def _finish(self, attn_output: torch.Tensor, out: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
        out.copy_(attn_output.view_as(out))
        if gate is not None:
            out.mul_(torch.sigmoid(gate).view_as(out))
        return self._into_folded_basis(out)

    def _columns(self, length: int) -> torch.Tensor:
        """``arange(length)`` for the mask, built once for the longest length asked for.

        A fresh ``arange`` per layer per step is a launch and an allocation for a
        tensor that is the same every time. One buffer, grown as needed and
        sliced, is neither -- and under graph capture the slice is a view of an
        address the replay already knows.
        """
        cached = getattr(self, "_column_index", None)
        if cached is None or cached.numel() < length:
            cached = torch.arange(max(length, self.geometry.max_seq_len), device=self.device)
            self._column_index = cached
        return cached[:length]

    def _prefix(self, layer: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The first ``length`` cached K and V as ``[length, H_kv, D]``, without a copy.

        The paging is the identity (a slot is a position), so the pool's first
        ``length`` rows *are* the prefix, in order.
        """
        packed = (-1, self.shape.num_kv_heads, self.shape.head_size)
        return (
            self.cache.key_planes[layer].view(packed)[:length],
            self.cache.value_planes[layer].view(packed)[:length],
        )


class NativeV5Backend(_DensePagedBackend):
    """The unquantised paged cache, through torch_npu's own attention.

    Which torch_npu entry point is ``attention_api``: by default the most preferred
    one this torch_npu has (``fia_v5`` where it exists). Whether the SoC accepts it
    is a runtime question -- Ascend 950 refuses ``fia_v1`` -- so ``smoke_glm``'s
    pre-flight tries the candidates on a synthetic layer and passes the one that
    worked in here.
    """

    name = "native_v5"

    def __init__(
        self,
        geometry: CacheGeometry,
        shape: LayerShape,
        device: torch.device,
        dtype: torch.dtype,
        output_rotation_folded: bool = False,
        attention_api: str | None = None,
    ) -> None:
        super().__init__(geometry, shape, device, dtype, output_rotation_folded)
        if attention_api is None:
            available = available_dense_attention_apis("prefill")
            if not available:
                raise RuntimeError(f"this torch_npu has none of {sorted(DENSE_ATTENTION_APIS.values())}")
            attention_api = available[0]
        if attention_api not in DENSE_ATTENTION_APIS:
            raise ValueError(f"unknown attention_api {attention_api!r}; choose one of {sorted(DENSE_ATTENTION_APIS)}")
        self.attention_api = attention_api
        # Built once: a prefill chunk passes it on every layer.
        self._causal_mask = torch.triu(
            torch.ones(_COMPRESSED_CAUSAL_MASK_SIZE, _COMPRESSED_CAUSAL_MASK_SIZE, dtype=torch.int8, device=device),
            diagonal=1,
        )

    def decode(
        self,
        layer: int,
        query: torch.Tensor,
        context_lens: torch.Tensor,
        out: torch.Tensor,
        gate: torch.Tensor | None = None,
        window: int | None = None,
    ) -> torch.Tensor:
        """Each query token a sequence of its own (TND), attending its whole context: no mask.

        ``window`` sizes the block table, and that is as far as it reaches here:
        both entry points take the kv lengths as a Python list, so they cross to
        the host whatever the window says. Hence ``supports_static_decode`` is
        left False -- this is the one backend a graph cannot be captured around,
        and saying so beats capturing one whose sequence lengths were frozen.
        """
        if self.attention_api not in _DECODE_CAPABLE_APIS:
            raise NotImplementedError(
                f"{self.attention_api} has no block table, so it cannot decode out of the paged cache; "
                f"it serves the dense_staging prefill only. Decode needs one of {list(_DECODE_CAPABLE_APIS)}."
            )
        num_tokens = query.shape[0]
        key, value = self.cache.flat(layer)
        block_tables = self._block_tables(num_tokens, window if window is not None else longest_context(context_lens))
        query_lens = list(range(1, num_tokens + 1))
        if self.attention_api == "fia_v5":
            attn_output, _ = _torch_npu().npu_fused_infer_attention_score_v2(
                query.contiguous(),
                key,
                value,
                block_table=block_tables,
                input_layout="TND",
                block_size=self.geometry.block_size,
                actual_seq_qlen=query_lens,
                actual_seq_kvlen=context_lens.tolist(),
                num_query_heads=self.shape.num_heads,
                num_key_value_heads=self.shape.num_kv_heads,
                softmax_scale=self.shape.scale,
            )
        else:
            attn_output, _ = _torch_npu().npu_fused_infer_attention_score(
                query=query.contiguous(),
                key=key,
                value=value,
                block_table=block_tables,
                input_layout="TND",
                block_size=self.geometry.block_size,
                actual_seq_lengths=query_lens,
                actual_seq_lengths_kv=context_lens.tolist(),
                num_key_value_heads=self.shape.num_kv_heads,
                num_heads=self.shape.num_heads,
                scale=self.shape.scale,
            )
        return self._finish(attn_output, out, gate)

    def prefill_chunk(
        self,
        layer: int,
        query: torch.Tensor,
        prefix_end: int,
        out: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One request of ``count`` query tokens against a ``prefix_end``-token cache.

        ``sparse_mode=3`` is the bottom-right causal mask: with more kv than q it
        aligns the diagonal to the *end* of the prefix, which is exactly a chunk
        arriving after ``prefix_end - count`` tokens are already cached. It reads
        the compressed 2048 x 2048 mask rather than one built per shape. Spelling
        the same mask out as one request per token (what the base class does)
        would be correct and far slower.
        """
        count = query.shape[0]
        if prefix_end - count < 0:
            raise ValueError(f"a {count}-token chunk cannot end at prefix {prefix_end}")
        if self.attention_api == "pfa":
            return self._finish(self._prompt_flash_attention(layer, query, prefix_end), out, gate)
        key, value = self.cache.flat(layer)
        block_table = self.cache.block_table(prefix_end)
        if self.attention_api == "fia_v5":
            attn_output, _ = _torch_npu().npu_fused_infer_attention_score_v2(
                query.contiguous(),
                key,
                value,
                atten_mask=self._causal_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=self.geometry.block_size,
                actual_seq_qlen=[count],
                actual_seq_kvlen=[prefix_end],
                num_query_heads=self.shape.num_heads,
                num_key_value_heads=self.shape.num_kv_heads,
                softmax_scale=self.shape.scale,
                sparse_mode=_BOTTOM_RIGHT_CAUSAL_SPARSE_MODE,
                next_tokens=_CAUSAL_NEXT_TOKENS,
            )
        else:
            attn_output, _ = _torch_npu().npu_fused_infer_attention_score(
                query=query.contiguous(),
                key=key,
                value=value,
                atten_mask=self._causal_mask,
                block_table=block_table,
                input_layout="TND",
                block_size=self.geometry.block_size,
                actual_seq_lengths=[count],
                actual_seq_lengths_kv=[prefix_end],
                num_key_value_heads=self.shape.num_kv_heads,
                num_heads=self.shape.num_heads,
                scale=self.shape.scale,
                sparse_mode=_BOTTOM_RIGHT_CAUSAL_SPARSE_MODE,
                next_tokens=_CAUSAL_NEXT_TOKENS,
            )
        return self._finish(attn_output, out, gate)

    def _prompt_flash_attention(self, layer: int, query: torch.Tensor, prefix_end: int) -> torch.Tensor:
        """PFA over the prefix as one contiguous ``[1, prefix_end, H_kv * D]`` sequence.

        :meth:`_prefix` in the layout PFA wants: the same rows, flattened over the
        kv heads and given the batch dimension, still without a copy.
        """
        count = query.shape[0]
        hidden_kv = self.shape.num_kv_heads * self.shape.head_size
        keys = self.cache.key_planes[layer].view(1, -1, hidden_kv)[:, :prefix_end]
        values = self.cache.value_planes[layer].view(1, -1, hidden_kv)[:, :prefix_end]
        return _torch_npu().npu_prompt_flash_attention(
            query.contiguous().view(1, count, -1),
            keys,
            values,
            atten_mask=self._causal_mask,
            actual_seq_lengths=[count],
            actual_seq_lengths_kv=[prefix_end],
            num_heads=self.shape.num_heads,
            num_key_value_heads=self.shape.num_kv_heads,
            scale_value=self.shape.scale,
            input_layout="BSH",
            sparse_mode=_BOTTOM_RIGHT_CAUSAL_SPARSE_MODE,
            next_tokens=_CAUSAL_NEXT_TOKENS,
        )


class CANNDenseBackend(_DensePagedBackend):
    """The unquantised fp16 cache, attended by discrete CANN operators.

    No fused attention kernel at all: the scores are a batched matmul, the
    softmax is a softmax, and the values are a second batched matmul. On NPU
    tensors torch_npu dispatches those to CANN's own ``aclnnBatchMatMul`` and
    ``aclnnSoftmax``, so the Cube and Vector units do the arithmetic and the host
    only launches -- but none of it goes through ``aclnnFusedInferAttentionScore``,
    which is the point. Ascend 950 refuses that operator's V1 to V4 interfaces
    (``EZ9903``) and op-plugin reaches no further than V4, so on a 950 every
    :class:`NativeV5Backend` entry point fails its pre-flight and this is the only
    unquantised baseline left. It is slower than a fused kernel by the margin a
    materialised score matrix costs; it is not a *reference*, because the operators
    under it are the device's.

    One code path serves both halves of the run, because both are the same
    question. :meth:`decode` gives query row ``r`` the whole of its own context
    (``context_lens[r]`` columns) and :meth:`prefill_chunk` gives it
    ``prefix_end - count + r + 1`` -- bottom-right causal written out as a column
    count. So both build a per-row visible length and hand it to :meth:`_attend`,
    which masks on it. The masking is exact, never an approximation of a causal
    kernel's: a column at or past a row's length contributes nothing.
    """

    name = "cann_dense"
    # Matmul, softmax, matmul: nothing here asks the host anything once the
    # window is fixed, and the cache write is a scatter at a fixed shape.
    supports_static_decode = True
    supports_graph_capture = True

    def decode(
        self,
        layer: int,
        query: torch.Tensor,
        context_lens: torch.Tensor,
        out: torch.Tensor,
        gate: torch.Tensor | None = None,
        window: int | None = None,
    ) -> torch.Tensor:
        """Each query row attends its whole context: visible length is the context length.

        Without a ``window`` the longest length is read back to the host, because
        how much of the pool to slice is then a host decision -- a device-to-host
        sync per layer per step. With one, the slice is the caller's constant and
        the lengths stay on the device, where :meth:`_attend`'s mask reads them.
        """
        lengths = context_lens.to(device=query.device, dtype=torch.int64).reshape(-1)
        longest = window if window is not None else longest_context(lengths)
        return self._finish(self._attend(layer, query, lengths, longest, checked=window is not None), out, gate)

    def prefill_chunk(
        self,
        layer: int,
        query: torch.Tensor,
        prefix_end: int,
        out: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The whole chunk in one pass, rather than the base class's row at a time.

        The default :meth:`AttentionBackend.prefill_chunk` re-presents a chunk as
        one decode per token because the TurboQuant kernels have no paged prefill.
        Here there is nothing to re-present: the visible lengths *are* the causal
        mask, so a 2048-token chunk is the same three launches a single token is.
        """
        count = query.shape[0]
        first = prefix_end - count
        if first < 0:
            raise ValueError(f"a {count}-token chunk cannot end at prefix {prefix_end}")
        lengths = torch.arange(first + 1, prefix_end + 1, dtype=torch.int64, device=query.device)
        return self._finish(self._attend(layer, query, lengths, prefix_end, checked=True), out, gate)

    def _attend(
        self, layer: int, query: torch.Tensor, lengths: torch.Tensor, longest: int, checked: bool = False
    ) -> torch.Tensor:
        """``query`` ``[q, H, D]`` over the prefix, row ``r`` seeing columns ``0 .. lengths[r] - 1``.

        The query heads sharing a kv head are folded into the batched matmul's rows,
        so each stage is one launch over ``H_kv`` rather than a loop over ``H``.
        The query is scaled before the first matmul, not the scores after it: at
        float16 an unscaled ``q . k`` over a long context is what overflows, and
        the scaled one cannot.

        Neither K nor V is copied to be multiplied. The pool holds ``[k, H_kv, D]``
        contiguous, so ``keys.permute(1, 2, 0)`` is ``[H_kv, D, k]`` with the D axis
        at stride 1 -- a column-major ``K``, which is exactly the transposed operand
        a batched GEMM takes a flag for -- and ``values.permute(1, 0, 2)`` is
        ``[H_kv, k, D]`` row-major. Both carry batch stride ``D`` over the kv heads.
        These are the layouts to keep if this is ever rewritten: a permute that
        landed either operand with no unit-stride axis would turn one view per
        layer per step into one full copy of the prefix.

        Tiled over the query rows only because the score matrix is materialised --
        a 2048-token chunk over a 32k prefix is 32 x 2048 x 32768 float32 scores,
        8 GB at once. The tiling changes the launch count, never the result.
        """
        if lengths.numel() != query.shape[0]:
            raise ValueError(f"{lengths.numel()} context lengths for {query.shape[0]} query rows")
        # An all-masked row would leave softmax dividing by zero and return NaN,
        # which nothing downstream raises on. Reading the shortest length back to
        # the host is the only way to notice, so it is skipped for a caller that
        # has already established the lengths are positive -- which both callers
        # that fix a window have: a decode's length is its position plus one, and
        # a prefill chunk's are ``arange(first + 1, ...)`` after ``first >= 0``.
        if not checked and lengths.numel() and int(lengths.min()) < 1:
            raise ValueError("a query row with a context length of 0 attends to nothing; there is no answer to return")
        if longest > self.geometry.max_seq_len:
            raise RuntimeError(f"context length {longest} exceeds the {self.geometry.max_seq_len}-token static cache")

        count, heads, head_size = query.shape
        kv_heads = self.shape.num_kv_heads
        group = heads // kv_heads
        keys, values = self._prefix(layer, longest)
        keys_t = keys.permute(1, 2, 0)  # [H_kv, D, k]
        values_t = values.permute(1, 0, 2)  # [H_kv, k, D]
        columns = self._columns(longest).view(1, 1, 1, -1)

        rows_per_tile = max(1, _SCORE_TILE_ELEMENTS // max(1, heads * longest))
        tiles = []
        for start in range(0, count, rows_per_tile):
            tile = query[start : start + rows_per_tile]
            rows = tile.shape[0]
            stacked = (tile * self.shape.scale).view(rows, kv_heads, group, head_size)
            stacked = stacked.permute(1, 2, 0, 3).reshape(kv_heads, group * rows, head_size)
            scores = torch.bmm(stacked, keys_t).float().view(kv_heads, group, rows, longest)
            visible = lengths[start : start + rows].view(1, 1, -1, 1)
            # In place: .float() above already returned a tensor of this tile's own,
            # and a second one of it is the largest allocation in the loop.
            scores.masked_fill_(columns >= visible, float("-inf"))
            weights = torch.softmax(scores, dim=-1).to(values.dtype).view(kv_heads, group * rows, longest)
            attended = torch.bmm(weights, values_t).view(kv_heads, group, rows, head_size)
            tiles.append(attended.permute(2, 0, 1, 3).reshape(rows, heads, head_size))
        return tiles[0] if len(tiles) == 1 else torch.cat(tiles)


class _TurboQuantBackend(AttentionBackend):
    """Shared plumbing for the two TurboQuant decodes: the cache, Pi, and the scratch."""

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
        layout, rotation = turboquant_layout(), turboquant_rotation()
        self._pi_signs = rotation.turboquant_pi_signs(shape.head_size, device)
        self._hadamard16 = layout.turboquant_hadamard16(device)
        self._write_tables = layout.turboquant_codec_tables(shape.head_size, _ROTATE_BATCH_ROWS, device)
        self._decode_tables = layout.turboquant_codec_tables(shape.head_size, layout.TURBOQUANT_TILE_ROWS, device)
        self._workspace: torch.Tensor | None = None
        self._workspace_floats: dict[tuple[int, int], int] = {}
        self._rotated_query: torch.Tensor | None = None

    @property
    @abc.abstractmethod
    def _writer(self):
        """The reshape_and_cache the matching decode reads back.

        The two decodes read different byte layouts out of the same packed
        planes, so the writer follows the decode. Pairing them wrongly does not
        raise -- it decodes one layout as the other.
        """

    def write_kv(self, layer: int, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor) -> None:
        key_plane, value_plane, scale_plane = self.cache.planes(layer)
        self._writer(
            key.contiguous(),
            value.contiguous(),
            key_plane,
            value_plane,
            scale_plane,
            slots.to(torch.int32).contiguous(),
            self._pi_signs,
            self._write_tables,
        )

    def _rotated_query_buffer(self, num_tokens: int) -> torch.Tensor:
        needed = num_tokens * self.shape.num_heads * self.shape.head_size
        buffer = self._rotated_query
        if buffer is None or buffer.numel() < needed:
            self._rotated_query = torch.empty(needed, dtype=torch.float32, device=self.device)
            buffer = self._rotated_query
        return buffer[:needed].view(num_tokens, self.shape.num_heads, self.shape.head_size)

    @abc.abstractmethod
    def _workspace_size(self, num_tokens: int, max_blocks_per_seq: int) -> int:
        """Ask the operator's own planner, rather than reproducing its arithmetic."""

    def _decode_workspace(self, num_tokens: int, max_blocks_per_seq: int) -> torch.Tensor:
        key = (num_tokens, max_blocks_per_seq)
        needed = self._workspace_floats.get(key)
        if needed is None:
            needed = self._workspace_size(num_tokens, max_blocks_per_seq)
            self._workspace_floats[key] = needed
        workspace = self._workspace
        if workspace is None or workspace.numel() < needed:
            # Not monotonic in num_tokens: a long context small enough to leave
            # cores idle is split further, so the peak can sit at a small decode.
            self._workspace = torch.empty(max(needed, 1), dtype=torch.float32, device=self.device)
            workspace = self._workspace
        return workspace


class TurboQuantCubeBackend(_TurboQuantBackend):
    """One kv4fp8 Cube launch per step: rotate, attend, un-rotate, gate, write.

    The launch is handed the raw fp16 query and changes basis itself, so nothing
    precedes it and nothing follows it -- which is the point, and what the
    output stage exists to make true for a gated layer as well.
    """

    name = "turboquant_cube"
    # The kernel takes the context lengths as a device tensor and the writer
    # takes the slots as one; with the block table's width fixed, a step asks
    # the host nothing. Capture is declared but has not been exercised on
    # silicon from here -- a device that refuses it falls back (decode_graph
    # 'auto'), which is what that mode is for.
    supports_static_decode = True
    supports_graph_capture = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.geometry.require_cube_tiling()
        # Only when the full extension has not registered it: a standalone build
        # (tools/tq_longbench/build_turboquant_ops.py), from $TURBOQUANT_LIB_PATH,
        # tools/tq_longbench/lib/ or build/.
        outcome = load_turboquant_library(cube_decode_available)
        if not outcome.registered:
            raise RuntimeError(
                "npu_turboquant_cube_decode is not registered: this build has no kv4fp8 Cube kernels. "
                "Use --backend turboquant_aiv, build with VLLM_ENABLE_TURBOQUANT_CUBE on an Ascend 950 target, or "
                f"build the standalone library with tools/tq_longbench/build_turboquant_ops.py. {outcome.detail}."
            )

    @property
    def fuses_output_gate(self) -> bool:
        """Only an unfolded layer can carry a gate: the gate sits where the output is still rotated."""
        return not self.output_rotation_folded

    @property
    def _writer(self):
        return ascend_ops().npu_turboquant_cube_reshape_and_cache

    def _workspace_size(self, num_tokens: int, max_blocks_per_seq: int) -> int:
        return int(
            ascend_ops().npu_turboquant_cube_workspace_size(
                num_tokens,
                self.shape.num_heads,
                self.shape.num_kv_heads,
                self.shape.head_size,
                max_blocks_per_seq,
                self.geometry.block_size,
            )
        )

    def _output_stage(self, gate: torch.Tensor | None):
        layout = turboquant_layout()
        if self.output_rotation_folded:
            if gate is not None:
                raise ValueError(
                    "a layer with Pi folded into o_proj cannot carry an output gate: the gate sits between "
                    "attention and o_proj, where the output is still rotated"
                )
            return layout.TurboQuantOutputStage.ROTATED_BASIS
        if gate is None:
            return layout.TurboQuantOutputStage.UNROTATED
        return layout.TurboQuantOutputStage.GATED

    def decode(
        self,
        layer: int,
        query: torch.Tensor,
        context_lens: torch.Tensor,
        out: torch.Tensor,
        gate: torch.Tensor | None = None,
        window: int | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        key_plane, value_plane, scale_plane = self.cache.planes(layer)
        block_tables = self._block_tables(num_tokens, window if window is not None else longest_context(context_lens))
        ascend_ops().npu_turboquant_cube_decode(
            query.contiguous(),
            None if gate is None else gate.contiguous(),
            self._pi_signs,
            self._write_tables,
            self._hadamard16,
            key_plane,
            value_plane,
            scale_plane,
            block_tables,
            context_lens.to(torch.int32).contiguous(),
            self._decode_workspace(num_tokens, block_tables.shape[1]),
            self._rotated_query_buffer(num_tokens),
            self.shape.num_kv_heads,
            self.shape.num_heads,
            self.shape.scale,
            int(self._output_stage(gate)),
            out,
        )
        return out


class TurboQuantAivBackend(_TurboQuantBackend):
    """The vector-unit decode: rotate the query, attend, un-rotate the output.

    Three launches where the Cube path takes one, and correspondingly slower --
    but it runs on every build and both activation dtypes, and it is the path
    the CPU stand-ins implement, so it is what the harness can hold to exact
    attention without a device.
    """

    name = "turboquant_aiv"
    # The kernel takes the context lengths as a device tensor and the writer
    # takes the slots as one; with the block table's width fixed, a step asks
    # the host nothing. Capture is declared but has not been exercised on
    # silicon from here -- a device that refuses it falls back (decode_graph
    # 'auto'), which is what that mode is for.
    supports_static_decode = True
    supports_graph_capture = True

    @property
    def _writer(self):
        return ascend_ops().npu_turboquant_reshape_and_cache

    def _workspace_size(self, num_tokens: int, max_blocks_per_seq: int) -> int:
        return int(
            ascend_ops().npu_turboquant_workspace_size(
                num_tokens,
                self.shape.num_heads,
                self.shape.head_size,
                max_blocks_per_seq,
                self.geometry.block_size,
            )
        )

    def decode(
        self,
        layer: int,
        query: torch.Tensor,
        context_lens: torch.Tensor,
        out: torch.Tensor,
        gate: torch.Tensor | None = None,
        window: int | None = None,
    ) -> torch.Tensor:
        ops = ascend_ops()
        num_tokens = query.shape[0]
        rotated = self._rotated_query_buffer(num_tokens)
        ops.npu_turboquant_rotate_q(query.contiguous(), self._pi_signs, self._write_tables, self._hadamard16, rotated)
        key_plane, value_plane, scale_plane = self.cache.planes(layer)
        block_tables = self._block_tables(num_tokens, window if window is not None else longest_context(context_lens))
        ops.npu_turboquant_paged_attention(
            rotated,
            key_plane,
            value_plane,
            scale_plane,
            block_tables,
            context_lens.to(torch.int32).contiguous(),
            self._decode_tables,
            self._decode_workspace(num_tokens, block_tables.shape[1]),
            self.shape.num_kv_heads,
            self.shape.num_heads,
            self.shape.scale,
            out,
        )
        if not self.output_rotation_folded:
            # The decode leaves O~ in the rotated basis; Pi is an involution, so
            # the same operator that rotates the query un-rotates the output.
            ops.npu_turboquant_rotate_q(out, self._pi_signs, self._write_tables, self._hadamard16, rotated)
            out.copy_(rotated.to(out.dtype))
        if gate is not None:
            out.mul_(torch.sigmoid(gate).view_as(out))
        return out


#: Backends that decode out of an unquantised pool. ``dense_staging`` needs one
#: of these for its prefill, and they are the only backends that can *decode*
#: out of it -- which is why asking for one with ``batched_decode`` is refused.
DENSE_BACKENDS = frozenset({"native_v5", "cann_dense", "dense_reference"})

#: The device backends, by ``--backend`` name.
ASCEND_BACKENDS = {
    NativeV5Backend.name: NativeV5Backend,
    CANNDenseBackend.name: CANNDenseBackend,
    TurboQuantCubeBackend.name: TurboQuantCubeBackend,
    TurboQuantAivBackend.name: TurboQuantAivBackend,
}

#: Of those, the ones that launch Ascend C operators from this repository (the
#: TurboQuant kernels) or the vendor's fused attention. They need an NPU -- or,
#: for the TurboQuant pair, a CPU with the stand-ins registered.
#: :class:`CANNDenseBackend` is not among them: matmul and softmax are torch
#: operators wherever it runs, so it is refused nowhere and is honest everywhere
#: about being the device's operators only when the device is one.
_ACCELERATOR_ONLY_BACKENDS = frozenset(ASCEND_BACKENDS) - {CANNDenseBackend.name}

#: What a ``--device`` with no Ascend runtime should use instead. The dense pair
#: and the quantised pair answer the same questions; only the second element of
#: each is the thing under test.
REFERENCE_EQUIVALENT = {
    NativeV5Backend.name: "dense_reference",
    CANNDenseBackend.name: "dense_reference",
    TurboQuantCubeBackend.name: "turboquant_reference",
    TurboQuantAivBackend.name: "turboquant_reference",
}


def backend_names() -> list[str]:
    """Every backend ``--backend`` accepts, Ascend and reference alike."""
    from tq_longbench.reference import REFERENCE_BACKENDS

    return sorted({*ASCEND_BACKENDS, *REFERENCE_BACKENDS})


def build_backend(
    name: str,
    geometry: CacheGeometry,
    shape: LayerShape,
    device: torch.device,
    dtype: torch.dtype,
    output_rotation_folded: bool = False,
    dense_attention_api: str | None = None,
) -> AttentionBackend:
    """Construct the named backend, allocating its cache pool.

    ``dense_attention_api`` picks :class:`NativeV5Backend`'s torch_npu entry point
    (see :data:`DENSE_ATTENTION_APIS`); every other backend ignores it.

    An Ascend backend asked for where its operators cannot run is **refused**
    rather than quietly substituted: the reference backends compute the same
    numbers but say nothing about what a kernel costs, and a latency table
    torch produced under an Ascend backend's name would be worse than no table.
    Use the ``*_reference`` names, or :func:`reference_equivalent`, to ask for
    them deliberately.

    "Cannot run" is about the operators, not the vendor. A CPU that has the
    stand-ins registered (:func:`tq_longbench.cpu_reference.cpu_turboquant_ops`)
    *can* serve them, and refusing there would lock the harness out of its own
    host tests; a CUDA device never can, and never will from this repository.
    :class:`CANNDenseBackend` has no such operators to refuse over -- matmul and
    softmax exist on every device -- so it builds anywhere, and is CANN-native
    only in the sense that on an NPU those are the operators torch_npu launches.
    """
    from tq_longbench.reference import REFERENCE_BACKENDS

    if name in REFERENCE_BACKENDS:
        if name == "dense_reference":
            return REFERENCE_BACKENDS[name](geometry, shape, device, dtype, output_rotation_folded)
        return REFERENCE_BACKENDS[name](geometry, shape, device, output_rotation_folded=output_rotation_folded)

    if name not in ASCEND_BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose one of {backend_names()}")
    # Before the device check: a dtype the kernel cannot take is wrong wherever
    # it is asked for, and saying so is more use than naming the device.
    if name == TurboQuantCubeBackend.name and dtype != torch.float16:
        raise ValueError(
            f"the Cube decode is float16 only, got {dtype}. Use --dtype float16, or --backend turboquant_aiv."
        )
    accelerator_only = name in _ACCELERATOR_ONLY_BACKENDS
    if accelerator_only and device.type != "npu" and not (device.type == "cpu" and _operators_served()):
        raise ValueError(
            f"{name} launches Ascend C operators, which cannot run on {device}. "
            f"Use --backend {REFERENCE_EQUIVALENT[name]} for the same arithmetic in torch "
            "(correctness only -- it says nothing about latency)."
        )
    if name == NativeV5Backend.name:
        return NativeV5Backend(geometry, shape, device, dtype, output_rotation_folded, dense_attention_api)
    if name == CANNDenseBackend.name:
        return CANNDenseBackend(geometry, shape, device, dtype, output_rotation_folded)
    return ASCEND_BACKENDS[name](geometry, shape, device, output_rotation_folded=output_rotation_folded)


def _operators_served() -> bool:
    """Whether something has registered the TurboQuant operators in this process."""
    ops = getattr(torch.ops, _ASCEND_NAMESPACE, None)
    return ops is not None and hasattr(ops, "npu_turboquant_reshape_and_cache")


def reference_equivalent(name: str) -> str:
    """The torch backend that computes what ``name`` computes."""
    return REFERENCE_EQUIVALENT.get(name, name)


def dense_backend_for(device: torch.device) -> str:
    """The unquantised backend this device can actually run.

    ``dense_staging`` prefill needs one whatever the decode is, so the choice
    follows the device rather than the ``--backend`` flag.
    """
    return dense_backend_candidates(device)[0]


def dense_backend_candidates(device: torch.device) -> list[str]:
    """The unquantised backends this device could stage a prefill through, best first.

    On an NPU the fused kernel comes first because it is the one worth measuring,
    and :class:`CANNDenseBackend` behind it because it is the one that still runs
    when op-plugin's FIA is refused -- which on Ascend 950 is always. Which of the
    two a run gets is not guessed: :func:`tq_longbench.preflight.select_dense_backend`
    tries them in this order on a synthetic layer before any weights load.
    """
    if device.type != "npu":
        return ["dense_reference"]
    return [NativeV5Backend.name, CANNDenseBackend.name]
