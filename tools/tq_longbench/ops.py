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
"""The three attention paths a run can take, and the buffers they reuse.

Each backend answers the same two questions -- write this chunk's K/V into the
cache, and attend over the prefix -- so the decoder layer never learns which one
it is driving.  What differs is underneath:

* :class:`NativeV5Backend` -- ``npu_fused_infer_attention_score`` over the dense
  paged cache.  The baseline, and the only path whose numbers are not a
  quantisation of anything.
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

from tq_longbench._ascend import turboquant_layout, turboquant_rotation
from tq_longbench.kv_cache import CacheGeometry, DenseKVCache, TurboQuantKVCache

# The codec table image is built for a batch of rows; the writer and the query
# rotation use one, the paged decode tiles 16 at a time.
_ROTATE_BATCH_ROWS = 1

_ASCEND_NAMESPACE = "_C_ascend"

# npu_fused_infer_attention_score's bottom-right causal mask: with more kv than
# q the diagonal aligns to the end of the prefix, which is what a chunk arriving
# after a populated cache needs.
_BOTTOM_RIGHT_CAUSAL_SPARSE_MODE = 3


def ascend_ops():
    """The operator namespace the extension registers into.

    Raised on rather than returned as ``None``: every caller needs it, and an
    ``AttributeError`` from deep inside a launch says far less than this does.
    """
    ops = getattr(torch.ops, _ASCEND_NAMESPACE, None)
    if ops is None or not hasattr(ops, "npu_turboquant_reshape_and_cache"):
        raise RuntimeError(
            "the TurboQuant operators are not registered. Build vllm-ascend's C++ extension with "
            "VLLM_ENABLE_TURBOQUANT (and VLLM_ENABLE_TURBOQUANT_CUBE for the Cube decode), or run under "
            "tests.ut.attention.turboquant_cpu_ops.turboquant_cpu_ops() to serve them from the CPU."
        )
    return ops


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


class AttentionBackend(abc.ABC):
    """One decode path, bound to one cache pool, serving every layer of the model."""

    name: str
    #: Whether :meth:`decode` applies ``sigmoid(gate)`` itself. When False the
    #: caller multiplies in torch, so a gated layer is correct either way.
    fuses_output_gate: bool = False
    #: Whether the ``o_proj`` this backend feeds has Pi folded into it, so the
    #: attention output must leave in the rotated basis, ``O Pi``.
    output_rotation_folded: bool = False

    def __init__(self, geometry: CacheGeometry, shape: LayerShape, device: torch.device) -> None:
        self.geometry = geometry
        self.shape = shape
        self.device = device

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
    ) -> torch.Tensor:
        """Attend ``query`` ``[num_tokens, num_heads, head_size]`` over each token's prefix.

        ``context_lens`` holds one length per query token, so the same call
        serves a single decode step and a whole prefill chunk presented as a
        batch of per-position decodes.
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
        return self.decode(layer, query, context_lens, out, gate)

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


class NativeV5Backend(AttentionBackend):
    """``npu_fused_infer_attention_score`` over the unquantised paged cache."""

    name = "native_v5"

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

    def decode(
        self,
        layer: int,
        query: torch.Tensor,
        context_lens: torch.Tensor,
        out: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        import torch_npu  # Deferred: only this backend needs the device runtime.

        num_tokens = query.shape[0]
        key, value = self.cache.flat(layer)
        block_table = self.cache.block_table(int(context_lens.max()) if context_lens.numel() else 0)
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query.contiguous(),
            key=key,
            value=value,
            block_table=block_table.expand(num_tokens, -1).contiguous(),
            input_layout="TND",
            block_size=self.geometry.block_size,
            actual_seq_lengths=list(range(1, num_tokens + 1)),
            actual_seq_lengths_kv=context_lens.tolist(),
            num_key_value_heads=self.shape.num_kv_heads,
            num_heads=self.shape.num_heads,
            scale=self.shape.scale,
        )
        out.copy_(attn_output.view_as(out))
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
        """One request of ``count`` query tokens against a ``prefix_end``-token cache.

        ``sparse_mode=3`` is the bottom-right causal mask the operator builds
        itself: with more kv than q it aligns the diagonal to the *end* of the
        prefix, which is exactly a chunk arriving after ``prefix_end - count``
        tokens are already cached.  Spelling the same mask out as one request per
        token (what the base class does) would be correct and far slower.
        """
        import torch_npu  # Deferred: only this backend needs the device runtime.

        count = query.shape[0]
        if prefix_end - count < 0:
            raise ValueError(f"a {count}-token chunk cannot end at prefix {prefix_end}")
        key, value = self.cache.flat(layer)
        block_table = self.cache.block_table(prefix_end)
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query.contiguous(),
            key=key,
            value=value,
            block_table=block_table,
            input_layout="TND",
            block_size=self.geometry.block_size,
            actual_seq_lengths=[count],
            actual_seq_lengths_kv=[prefix_end],
            num_key_value_heads=self.shape.num_kv_heads,
            num_heads=self.shape.num_heads,
            scale=self.shape.scale,
            sparse_mode=_BOTTOM_RIGHT_CAUSAL_SPARSE_MODE,
        )
        out.copy_(attn_output.view_as(out))
        if gate is not None:
            out.mul_(torch.sigmoid(gate).view_as(out))
        return self._into_folded_basis(out)


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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.geometry.require_cube_tiling()
        if not cube_decode_available():
            raise RuntimeError(
                "npu_turboquant_cube_decode is not registered: this build has no kv4fp8 Cube kernels. "
                "Use --backend turboquant_aiv, or build with VLLM_ENABLE_TURBOQUANT_CUBE on an Ascend 950 target."
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
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        key_plane, value_plane, scale_plane = self.cache.planes(layer)
        block_table = self.cache.block_table(int(context_lens.max()) if context_lens.numel() else 0)
        block_tables = block_table.expand(num_tokens, -1).contiguous()
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
    ) -> torch.Tensor:
        ops = ascend_ops()
        num_tokens = query.shape[0]
        rotated = self._rotated_query_buffer(num_tokens)
        ops.npu_turboquant_rotate_q(query.contiguous(), self._pi_signs, self._write_tables, self._hadamard16, rotated)
        key_plane, value_plane, scale_plane = self.cache.planes(layer)
        block_table = self.cache.block_table(int(context_lens.max()) if context_lens.numel() else 0)
        block_tables = block_table.expand(num_tokens, -1).contiguous()
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
DENSE_BACKENDS = frozenset({"native_v5", "dense_reference"})

#: Backends that launch Ascend C operators. They need the extension and an NPU.
ASCEND_BACKENDS = {
    NativeV5Backend.name: NativeV5Backend,
    TurboQuantCubeBackend.name: TurboQuantCubeBackend,
    TurboQuantAivBackend.name: TurboQuantAivBackend,
}

#: What a ``--device`` with no Ascend runtime should use instead. The dense pair
#: and the quantised pair answer the same questions; only the second element of
#: each is the thing under test.
REFERENCE_EQUIVALENT = {
    NativeV5Backend.name: "dense_reference",
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
) -> AttentionBackend:
    """Construct the named backend, allocating its cache pool.

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
    if device.type != "npu" and not (device.type == "cpu" and _operators_served()):
        raise ValueError(
            f"{name} launches Ascend C operators, which cannot run on {device}. "
            f"Use --backend {REFERENCE_EQUIVALENT[name]} for the same arithmetic in torch "
            "(correctness only -- it says nothing about latency)."
        )
    if name == NativeV5Backend.name:
        return NativeV5Backend(geometry, shape, device, dtype, output_rotation_folded)
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
    return NativeV5Backend.name if device.type == "npu" else "dense_reference"
