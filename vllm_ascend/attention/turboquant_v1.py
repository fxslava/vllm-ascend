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
"""TurboQuant 4-bit KV cache attention backend for Ascend 950PR and 910B.

The cache is plain contiguous ND -- ``(2, num_blocks, block_size, num_kv_heads,
head_size // 2)`` int8, two 4-bit codes per byte.

``q``, ``k`` and ``v`` reach the operators holding exactly the values the
projections and RoPE produced -- packed contiguous, because the adapter refuses
the strided slices a fused QKV split leaves -- and the kernels apply

    Pi x = D (H (D x)),   D = diag(+-1),   H = normalised Walsh-Hadamard

in UB.  Pi is symmetric (``Pi^T = D^T H^T D^T = D H D = Pi``) and an involution
(``Pi^2 = D H D D H D = I``).  Being orthogonal, it leaves scores alone:
``(Pi q) . (Pi k) == q . k``.  Keeping Q, K and V rotation out of the weights is
what makes RoPE correct: RoPE is applied *before* anything is rotated.

The decode's output is left in the rotated basis, ``O~ = softmax(q k^T) V~``:
the combine kernel no longer un-rotates it.  Where it goes back out depends on
the layer, and :attr:`AscendTurboQuantAttentionBackendImpl.output_rotation_folded`
records which:

* folded -- the checkpoint's ``o_proj`` was rewritten offline to
  ``W_o (I_H (x) Pi)`` (``scripts/tq_fold_output_rotation.py``), so the
  projection un-rotates for free.  Prefill then has to produce ``O~`` as well,
  and does by rotating V before the dense attention.
* not folded -- the default, and the only option for a layer with an
  elementwise output gate between attention and ``o_proj``.  The decode
  un-rotates its own output on the device, and prefill is unchanged.

See :mod:`vllm_ascend.attention.turboquant_rotation` for the fold itself.

Two decodes serve the cache, and a layer picks one when its impl is built,
because they write different byte layouts into the same packed planes:

* the Cube decode (:attr:`AscendTurboQuantAttentionBackendImpl.cube_decode`;
  Ascend 950, float16, ``VLLM_ASCEND_TURBOQUANT_CUBE_DECODE``) -- a kv4fp8
  NZ-tiled cache, and ONE launch per step that rotates the raw query and runs
  the output stage (:class:`TurboQuantOutputStage`) before its fp16 write.  An
  unfolded layer's output is un-rotated there, and a gated layer's is also
  multiplied by ``sigmoid(gate)`` when the model hands the gate over through
  ``vllm::turboquant_gated_attention`` (:attr:`fuses_output_gate`);
* the AIV decode -- every other build and dtype: ``npu_turboquant_rotate_q``,
  the paged decode, and ``npu_turboquant_rotate_q`` again over the output
  unless the layer is folded; a gate handed over is applied in torch.

Graph mode: a TurboQuant run is compiled and captured *piecewise*.  Both entry
points into this file -- ``vllm::unified_attention_with_output`` and
``vllm::turboquant_gated_attention`` -- are graph splitting ops, so the model
around attention is captured while these kernels are launched eagerly between
the pieces, and nothing here is ever traced under FakeTensor.
``NPUPlatform._apply_turboquant_graph_mode`` falls back to piecewise from a full
graph, which used to be a hard requirement: the decode staged the host-side
sequence lengths onto the device per step, and a captured copy would replay the
host address it recorded.  It no longer does -- it reads the runner's own
persistent device buffer (:meth:`_decode_seq_lens`) -- so the fallback is now a
statement about what has been verified on hardware rather than about what the
decode can express.  Everything persistent
below -- the scale plane, the reduction workspace, the rotated query, the index
staging buffers -- is sized to the configured ceiling on its first allocation
and refuses to grow inside a capture, so a step between two captured pieces
never moves an address a piece was captured against.
"""

import torch
import torch_npu
from vllm.logger import logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionLayer, AttentionType  # type: ignore

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendMetadata,
)

# Re-exported: the cache geometry and constant tables live in a module with no vLLM
# import, so tools/tq_longbench can load them by file path and drive these same
# operators with no engine around them. See turboquant_layout for why that matters.
from vllm_ascend.attention.turboquant_layout import (  # noqa: F401  (re-export)
    TURBOQUANT_BURST_FLOATS,
    TURBOQUANT_CUBE_TILE_ROWS,
    TURBOQUANT_LEVELS,
    TURBOQUANT_LLOYD_MAX_CENTROIDS,
    TURBOQUANT_LLOYD_MAX_THRESHOLDS,
    TURBOQUANT_PACK_FACTOR,
    TURBOQUANT_ROTATE_TILE,
    TURBOQUANT_TILE_ROWS,
    TurboQuantOutputStage,
    turboquant_codec_tables,
    turboquant_hadamard16,
    turboquant_scale_slot,
)
from vllm_ascend.attention.turboquant_rotation import output_rotation_is_folded, turboquant_pi_signs

# Uncompressed attention sinks: the side-car plane, its ingestion, and the host-side
# softmax merge that folds it back into a decode. Inert unless VLLM_ASCEND_TQ_SINK_TOKENS
# is set -- see turboquant_sink for the layout and for what the merge still costs.
from vllm_ascend.attention.turboquant_sink import (
    TurboQuantSinkCache,
    TurboQuantSinkConfig,
    fuse_decode_sinks,
)

# The diagnostic capture layer: what the operators were handed, appended to its own
# line-buffered file, and the dry run that walks the engine past the launches so a
# whole run's configuration can be recorded rather than the first faulting step's.
from vllm_ascend.attention.turboquant_trace import (
    cache_planes,
    describe_indices,
    describe_tensors,
    turboquant_tracer,
)
from vllm_ascend.attention.utils import notify_kv_cache_written

_WORKSPACE_MEMO_LIMIT = 1024

# Every global-memory copy the writers make -- the staged activations, the two packed
# planes and the scale plane -- moves whole 32-byte bursts, so an operand whose address
# is not a multiple of one starts the copy mid-burst. The AI core reports that as "the
# address for scalar to access GM is invalid" (error 264), not as a misalignment, which
# is why it is worth naming here rather than reading out of a core dump.
TURBOQUANT_GM_BURST_BYTES = TURBOQUANT_BURST_FLOATS * 4

# Every index operand the kernels take -- slot_mapping, block_tables, context_lens -- is read
# as `__gm__ int32_t *`. vLLM hands several of them over as int64, so the staging in
# _device_index narrows them on the host before the transfer rather than across it.
TURBOQUANT_INDEX_DTYPE = torch.int32


def turboquant_cube_decode_available() -> bool:
    """Whether this build registered the kv4fp8 Cube kernels (Ascend 950 builds only)."""
    return hasattr(torch.ops._C_ascend, "npu_turboquant_cube_decode")


def _model_dtype(impl: object) -> torch.dtype | None:
    vllm_config = getattr(impl, "vllm_config", None)
    model_config = getattr(vllm_config, "model_config", None)
    return getattr(model_config, "dtype", None)


def turboquant_cube_decode_selected(model_dtype: torch.dtype | None) -> bool:
    """Whether a layer decodes on the Cube: enabled, built, and float16 activations."""
    return (
        envs_ascend.VLLM_ASCEND_TURBOQUANT_CUBE_DECODE
        and model_dtype == torch.float16
        and turboquant_cube_decode_available()
    )


def _is_capturing() -> bool:
    """Whether an ACL graph capture is in progress *on this stream, right now*.

    Reading the flag needs a live forward context, and the answer is only ever
    used to refuse an allocation that would be unsafe *inside* a capture. Where
    there is no forward context at all there is no capture either, so a missing
    one is answered rather than raised -- that keeps a diagnostic from turning
    into the failure it exists to describe.

    The forward context's flag alone is not that answer. ``ACLGraphWrapper`` sets
    ``forward_context.capturing`` to True before it captures a piece and never
    clears it; the next step's ``set_ascend_forward_context`` does. Under piecewise
    compilation the attention layers are *split out* of those pieces, so from the
    first captured piece onwards every TurboQuant call in that forward sees the
    flag set while running perfectly ordinarily outside any capture -- and would
    refuse to grow a buffer it is entirely free to grow.

    So the flag is treated as the cheap necessary condition it is, and the stream
    is asked to confirm. The query is only reached when the flag is already set,
    which keeps the device out of the hot path and out of the CPU test harness.
    """
    try:
        if not _EXTRA_CTX.capturing:
            return False
    except (AssertionError, AttributeError, RuntimeError):
        return False

    is_stream_capturing = getattr(getattr(torch, "npu", None), "is_current_stream_capturing", None)
    if is_stream_capturing is None:
        # No way to ask: believe the flag rather than allocate into a capture.
        return True
    try:
        return bool(is_stream_capturing())
    except RuntimeError:
        return True


def _attention_phase(attn_metadata: AscendMetadata | None) -> str:
    """The batch's state, as a name a trace record can carry.

    Read defensively: the phase is only ever used to label a diagnostic, so a
    metadata object that does not carry a recognisable state is described rather
    than raised about -- the record exists precisely for the runs where something
    upstream is not what it should be.
    """
    if attn_metadata is None:
        return "no_metadata"
    state = getattr(attn_metadata, "attn_state", None)
    if state is None:
        return "unknown"
    if state == AscendAttentionState.DecodeOnly:
        return "decode"
    if state == AscendAttentionState.PrefillNoCache:
        return "prefill"
    return str(getattr(state, "name", state))


class AscendTurboQuantAttentionBackend(AscendAttentionBackend):
    """Backend descriptor for the 4-bit TurboQuant KV cache."""

    @staticmethod
    def get_name() -> str:
        return "ASCEND_TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type["AscendTurboQuantAttentionBackendImpl"]:
        return AscendTurboQuantAttentionBackendImpl

    @classmethod
    def supports_pcp(cls) -> bool:
        return False

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size // TURBOQUANT_PACK_FACTOR)

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [128]

    @staticmethod
    def swap_blocks(
        src_kv_cache: list[torch.Tensor],
        dst_kv_cache: list[torch.Tensor],
        src_to_dst: torch.Tensor,
    ) -> None:
        """Move packed blocks *and* the scale lanes that decode them.

        The base implementation moves ``kv_cache[0]`` and ``kv_cache[1]`` only.
        A TurboQuant block is meaningless without its scales, so relocating the
        codes without them does not raise -- it produces plausible wrong
        numbers. Anything the cache holds beyond the two packed planes is
        indexed by block in dimension 0 as well, so the same gather applies.
        """
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]
        for src_plane, dst_plane in zip(src_kv_cache, dst_kv_cache):
            if src_plane is None or dst_plane is None:
                continue
            dst_plane[dst_indices] = src_plane[src_indices].to(dst_plane.device)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        """Copy packed blocks together with their scale lanes.

        See :meth:`swap_blocks`: a copy that leaves the scales behind is the
        same silent corruption.
        """
        src_indices = src_to_dists[:, 0]
        dst_indices = src_to_dists[:, 1]
        for kv_cache in kv_caches:
            for plane in kv_cache:
                if plane is None:
                    continue
                plane[dst_indices] = plane[src_indices]


class AscendTurboQuantAttentionBackendImpl(AscendAttentionBackendImpl):
    """Attention over a 4-bit TurboQuant KV cache.

    Activate it on an existing layer with :func:`activate_turboquant_backend`,
    which swaps the impl class the same way the C8 KV cache scheme does.

    Scope: ``PrefillNoCache`` runs at full precision over the K/V of the current
    batch (nothing is read back from the cache), and ``DecodeOnly`` runs the
    TurboQuant paged kernel.  Chunked prefill and prefill-with-cache-hit would
    need a paged prefill kernel and are rejected explicitly rather than silently
    falling back to a path that cannot read this cache layout.
    """

    # Class-level defaults, so an impl whose class was swapped in after construction
    # (activate_turboquant_backend) starts on the AIV decode until it is told otherwise.
    cube_decode = False
    output_rotation_folded = False

    # Class-level for the same reason cube_decode is: activate_turboquant_backend swaps the
    # class of an already-built impl, so every attribute the methods below read has to exist
    # before __init__ would have set it. Sinks off is the shipping configuration.
    sink_config = TurboQuantSinkConfig()
    sink_cache = None

    # Which layer this is and how many times it has been entered, for the trace records.
    # Class-level for the same reason as above, and because everything downstream of
    # `forward` reads them: a record written from the cache writer has to name the layer
    # even though only `forward` is handed one.
    _trace_layer_name: str | None = None
    _trace_step = 0
    _trace_phase = "unknown"

    # Staging for an index operand the metadata keeps on the host; see _device_index.
    # None rather than {} because a mutable class attribute would be shared by every
    # layer; the dict is created per impl on first use.
    _device_index_buffers: dict[str, torch.Tensor] | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale_cache: torch.Tensor | None = None
        self._device_index_buffers = None
        self._pi_signs: torch.Tensor | None = None
        self.decode_workspace: torch.Tensor | None = None
        self._workspace_floats: dict[tuple[int, int], int] = {}
        self.rotated_query: torch.Tensor | None = None
        self._hadamard16: torch.Tensor | None = None
        self.output_rotation_folded = False
        self.cube_decode = turboquant_cube_decode_selected(_model_dtype(self))
        self.sink_config = TurboQuantSinkConfig.from_env()
        self.sink_cache: TurboQuantSinkCache | None = None

    @property
    def fuses_output_gate(self) -> bool:
        """Whether a gated layer should hand this impl its gate (vllm::turboquant_gated_attention).

        Only the Cube decode gates inside its launch, and only an unfolded layer
        can carry a gate at all: an elementwise gate between attention and
        o_proj is what rules the fold out.
        """
        return self.cube_decode and not self.output_rotation_folded

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        super().process_weights_after_loading(act_dtype)
        torch.ops._C_ascend.npu_turboquant_vector_core_num()
        logger.info_once(
            "[vllm-ascend/turboquant] 4-bit KV cache active; Pi is applied at runtime to K/V on the write path and "
            "to Q before each decode. The decode output stays rotated for a folded o_proj and is un-rotated on "
            "the device otherwise."
        )
        if self.cube_decode:
            logger.info_once(
                "[vllm-ascend/turboquant] kv4fp8 Cube decode: the query rotation, the output un-rotation and any "
                "attn_output_gate run inside the decode launch."
            )
        tracer = turboquant_tracer()
        if tracer.capture:
            logger.warning_once(
                "[vllm-ascend/turboquant] TURBOQUANT_CAPTURE_SIGNATURES=1: every operator invocation is appended "
                "to %s, at the cost of a device-to-host copy of the index tensors per call. Diagnostic only.",
                tracer.path,
            )
        if tracer.dry_run:
            logger.warning_once(
                "[vllm-ascend/turboquant] TURBOQUANT_DRY_RUN=1: no TurboQuant kernel will be launched and the "
                "attention output is zeroed. The run produces no meaningful tokens."
            )

    def pi_signs(self, device: torch.device) -> torch.Tensor:
        if self._pi_signs is None:
            self._pi_signs = turboquant_pi_signs(self.head_size, device)
        return self._pi_signs

    def codec_tables(self, device: torch.device, batch_rows: int) -> torch.Tensor:
        return turboquant_codec_tables(self.head_size, batch_rows, device)

    def hadamard16(self, device: torch.device) -> torch.Tensor:
        if self._hadamard16 is None:
            self._hadamard16 = turboquant_hadamard16(device)
        return self._hadamard16

    def _ensure_scale_cache(self, kv_cache: tuple[torch.Tensor, ...]) -> None:
        """Bind the fp32 scale plane that accompanies the packed cache.

        The packed cache shape is fixed by ``get_kv_cache_shape`` and has no room
        for a per-vector scale, so the scales occupy their own plane shaped
        ``(num_blocks, block_size, scale_slot)``.  Indexing by token rather than
        by head is what makes the kernel's scatter a single aligned burst; see
        :func:`turboquant_scale_slot` for what the padding costs.

        The model runner carves that plane out of the same KV cache allocation
        as the two packed planes and hands it over as ``kv_cache[2]``, budgeted
        by :class:`AscendTurboQuantAttentionSpec`.  That is what keeps it
        visible to everything that accounts for the cache in pages.

        A two-element ``kv_cache`` means the runner did not provide one, which
        is the pre-spec arrangement and still what the CPU harness passes.  The
        plane is then allocated here, once, and refused during a graph capture
        for the same reason the decode workspace is: a capture replays the
        addresses it recorded.
        """
        if self.scale_cache is not None:
            return

        num_blocks, block_size, num_kv_heads, _ = kv_cache[0].shape
        if self.cube_decode and block_size % TURBOQUANT_CUBE_TILE_ROWS != 0:
            raise ValueError(
                f"[vllm-ascend/turboquant] the Cube decode reads {TURBOQUANT_CUBE_TILE_ROWS}-row tiles, so the "
                f"kernel block size must be a multiple of {TURBOQUANT_CUBE_TILE_ROWS}, got {block_size}. "
                "Set VLLM_ASCEND_TURBOQUANT_CUBE_DECODE=0 to decode this cache on the AIV path instead."
            )
        shape = (num_blocks, block_size, turboquant_scale_slot(num_kv_heads))

        if len(kv_cache) >= 3 and kv_cache[2] is not None:
            provided = kv_cache[2]
            if provided.shape != shape:
                raise RuntimeError(
                    f"[vllm-ascend/turboquant] the scale plane is {tuple(provided.shape)}, expected {shape} "
                    f"for a {num_blocks}-block cache of {num_kv_heads} kv heads"
                )
            if provided.dtype != torch.float32:
                raise RuntimeError(f"[vllm-ascend/turboquant] the scale plane must be float32, got {provided.dtype}")
            self.scale_cache = provided
            return

        if _is_capturing():
            raise RuntimeError(
                f"[vllm-ascend/turboquant] the {shape} scale plane would have to be allocated during a "
                "graph capture. Run this layer once outside capture, or let the model runner carve the "
                "plane from the KV cache allocation so there is nothing left to allocate here."
            )
        self.scale_cache = torch.zeros(shape, dtype=torch.float32, device=kv_cache[0].device)

    def _ensure_sink_cache(self, kv_cache: tuple[torch.Tensor, ...], dtype: torch.dtype) -> None:
        """Bind the uncompressed sink plane, once, if the feature is on.

        Sized from the packed cache it accompanies and indexed by the same physical block,
        so it is allocated here rather than budgeted by the KV cache spec: the two packed
        planes and the scale plane are what the paged allocator accounts for, and this one
        is deliberately outside that accounting because it is not paged -- a block's sink
        row is written once, when the sequence that anchors the block is ingested.

        Refused during a graph capture for the reason :meth:`_ensure_scale_cache` refuses:
        a replay owns the addresses it was captured against.
        """
        if not self.sink_config.enabled or self.sink_cache is not None:
            return
        num_blocks, block_size, num_kv_heads, _ = kv_cache[0].shape
        if _is_capturing():
            raise RuntimeError(
                "[vllm-ascend/turboquant] the uncompressed sink plane would have to be allocated during a "
                "graph capture. Run this layer once outside capture, or unset VLLM_ASCEND_TQ_SINK_TOKENS."
            )
        self.sink_cache = TurboQuantSinkCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_sink_tokens=self.sink_config.num_sink_tokens,
            num_kv_heads=num_kv_heads,
            head_size=self.head_size,
            dtype=dtype,
            device=kv_cache[0].device,
        )
        logger.info_once(
            "[vllm-ascend/turboquant] VLLM_ASCEND_TQ_SINK_TOKENS=%d: the first %d tokens of each sequence are "
            "kept uncompressed in a %.1f MiB side-car plane (%d bytes per sequence per layer) and merged into "
            "the decode's softmax on the host. The merge recomputes the quantised context's softmax denominator, "
            "which neither decode operator returns, so it costs O(context) per step: diagnostic and accuracy "
            "work, not a serving default.",
            self.sink_config.num_sink_tokens,
            self.sink_config.num_sink_tokens,
            self.sink_cache.nbytes / (1 << 20),
            self.sink_cache.bytes_per_sequence,
        )

    def _ingest_sinks(self, key: torch.Tensor, value: torch.Tensor, attn_metadata: AscendMetadata) -> int:
        """Store each sequence's leading tokens uncompressed, beside the write that packed them.

        Runs on every cache write, not only on prefill: a sequence shorter than the sink
        count is still filling its sinks a decode token at a time, and the ingestion is
        driven by each token's position in its *sequence*. Once every sequence in the batch
        is past its sinks -- which is every steady-state decode step -- this finds nothing
        to copy and returns without touching the device.
        """
        ends = list(attn_metadata.actual_seq_lengths_q or ())
        if not ends or self.sink_cache is None:
            return 0
        num_actual_tokens = attn_metadata.num_actual_tokens
        starts = [0, *ends[:-1]]
        # Host tensors on purpose (see _decode_seq_lens): reading them costs no
        # synchronisation, and the ingestion is driven entirely by these indices.
        seq_lens = attn_metadata.seq_lens_list or attn_metadata.seq_lens.tolist()
        query_starts, query_lens, context_lens = [], [], []
        for request, (start, end) in enumerate(zip(starts, ends)):
            # A request that contributes nothing is described rather than dropped: the
            # anchor blocks are handed over as one row per request, so a gap in these lists
            # would shift every later request onto the wrong sequence's block. The FIA TND
            # layout leaves exactly such a request -- a dummy past the last real token, which
            # reshape_and_cache does not write and neither does this.
            length = max(0, min(end, num_actual_tokens) - start)
            described = request < len(seq_lens)
            query_starts.append(start)
            query_lens.append(length if described else 0)
            context_lens.append(max(0, int(seq_lens[request]) - length) if described else 0)
        if not any(query_lens):
            return 0
        return self.sink_cache.ingest(
            key,
            value,
            anchor_blocks=attn_metadata.block_tables[: len(ends), 0],
            query_starts=query_starts,
            query_lens=query_lens,
            context_lens=context_lens,
        )

    def _fuse_decode_sinks(
        self,
        attention_output: torch.Tensor,
        query: torch.Tensor,
        rotated_query: torch.Tensor,
        attn_metadata: AscendMetadata,
    ) -> None:
        """Replace the quantised sinks in a decode's output with the uncompressed ones.

        ``attention_output`` is still in the rotated basis here, which is what both decodes
        are asked to produce while this is on: the un-rotation and any output gate are
        linear in the output and are applied after the merge, not before it.
        """
        if self.sink_cache is None:
            return
        fuse_decode_sinks(
            attention_output=attention_output,
            query=query,
            rotated_query=rotated_query,
            sink_cache=self.sink_cache,
            key_cache=self.key_cache,
            value_cache=self.value_cache,
            scale_cache=self.scale_cache,
            block_tables=self._device_index(attn_metadata.block_tables, query.device, "block_tables"),
            seq_lens=attn_metadata.seq_lens,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale_value=self.scale,
            pi_signs=self.pi_signs(query.device),
        )

    def _unrotate_output(self, attention_output: torch.Tensor, rotated_query: torch.Tensor) -> None:
        """``O = Pi O~`` on the device, through the rotation operator and its scratch."""
        torch.ops._C_ascend.npu_turboquant_rotate_q(
            attention_output,
            self.pi_signs(attention_output.device),
            self.codec_tables(attention_output.device, 1),
            self.hadamard16(attention_output.device),
            rotated_query,
        )
        attention_output.copy_(rotated_query)

    def _validate_cache_write(self, key: torch.Tensor, value: torch.Tensor, slots: torch.Tensor) -> None:
        """Hold one cache write to what the kernels can address, before the launch.

        Off by default: the slot values live on the device, so reading them costs a
        device-to-host copy and a synchronisation on every layer of every step. Turn it
        on with ``VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS=1`` when a write faults.

        The writers already drop a slot they cannot address -- a negative one is how
        vLLM marks a padding token, and one past the last row of the cache is dropped
        beside it so a caller bug cannot scribble on whatever else owns that address.
        Dropping is silent, though, and a dropped row means a token whose K/V is simply
        not in the cache. This is what turns both into something named: an out-of-range
        slot raises, and a negative one inside the batch's own token range is warned
        about, since ``slot_mapping`` was already sliced to ``num_actual_tokens``.

        The pointer checks are here rather than in the operator because the operator is
        handed addresses, not tensors: ``.contiguous()`` returns a contiguous view
        unchanged, storage offset and all, so an operand can arrive contiguous and still
        start part-way through a burst.
        """
        operands = {
            "key": key,
            "value": value,
            "key_cache": self.key_cache,
            "value_cache": self.value_cache,
            "scale_cache": self.scale_cache,
            "slot_mapping": slots,
        }
        detail = ", ".join(
            f"{name}{tuple(tensor.shape)} {tensor.dtype} stride={tuple(tensor.stride())} @{tensor.data_ptr():#x}"
            for name, tensor in operands.items()
        )
        logger.debug("[vllm-ascend/turboquant] cache write operands: %s", detail)

        for name, tensor in operands.items():
            if not tensor.is_contiguous():
                raise RuntimeError(
                    f"[vllm-ascend/turboquant] {name} reaches the cache writer non-contiguous; the kernel "
                    f"indexes it as packed global memory. Operands: {detail}"
                )
            if tensor.data_ptr() % TURBOQUANT_GM_BURST_BYTES:
                raise RuntimeError(
                    f"[vllm-ascend/turboquant] {name} starts at {tensor.data_ptr():#x}, which is not a whole "
                    f"{TURBOQUANT_GM_BURST_BYTES}-byte burst; the writer's copies would start mid-burst. "
                    f"Operands: {detail}"
                )

        if slots.numel() == 0:
            return
        # The one place this costs a synchronisation, and the reason it is opt-in.
        lowest = int(slots.min())
        highest = int(slots.max())
        num_slots = self.key_cache.shape[0] * self.key_cache.shape[1]
        if highest >= num_slots:
            raise RuntimeError(
                f"[vllm-ascend/turboquant] slot_mapping holds {highest}, past the last row of a "
                f"{self.key_cache.shape[0]}-block cache of {self.key_cache.shape[1]}-token blocks "
                f"({num_slots} rows). The writer drops it rather than writing outside the cache, so "
                f"those tokens are missing from the cache. Operands: {detail}"
            )
        if lowest < 0:
            logger.warning_once(
                "[vllm-ascend/turboquant] slot_mapping carries %d inside the batch's own %d tokens; "
                "the writer drops those rows, so their K/V never reaches the cache.",
                lowest,
                slots.numel(),
            )

    def _trace_caller_context(self) -> dict:
        """Who is calling, and with which of this backend's two decodes.

        Every record carries it, because a trace is only readable if a record can
        be attributed to a layer and a step: the interesting failures are the ones
        where layer 0 step 1 is fine and layer 7 step 300 is not.
        """
        return {
            "layer": self._trace_layer_name,
            "step": self._trace_step,
            "phase": self._trace_phase,
            "is_prefill": self._trace_phase == "prefill",
            "is_decode": self._trace_phase == "decode",
            "decode_path": "cube" if self.cube_decode else "aiv",
            "output_rotation_folded": self.output_rotation_folded,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_size,
            "scale": float(self.scale),
        }

    def _cache_geometry(self) -> dict:
        """The bound cache's page geometry and the row capacity it implies.

        ``num_blocks * block_size`` is what a slot mapping is bounded by, and the
        number the writers silently drop a row for exceeding.
        """
        cache = self.key_cache
        if cache is None:
            return {"num_blocks": None, "block_size": None, "capacity": None}
        num_blocks, block_size = int(cache.shape[0]), int(cache.shape[1])
        return {"num_blocks": num_blocks, "block_size": block_size, "capacity": num_blocks * block_size}

    def _fence_cache_write(self, device: torch.device) -> None:
        """Drain the writer's launch so an async fault cannot be blamed on a later one.

        Ascend launches are asynchronous. A kernel that traps does not fail its own
        launch: the runtime reports the error from a *subsequent* launch API call, and
        names whichever operator the launch queue was working on. So a fault in this
        writer surfaces at some unrelated operator, and a fault in an unrelated operator
        surfaces at this writer -- which is why an "aclnnInplaceCopy" in the message does
        not mean ``write`` called one. It does not.

        The ``int(slots.min())`` in :meth:`_validate_cache_write` already drains
        everything queued before the write; this drains the write itself. Together they
        bracket it, so the next failure names the right side of the boundary. Diagnostic
        only, and part of what ``VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS`` costs. The
        global ``ASCEND_LAUNCH_BLOCKING=1`` does the same thing for every operator, but
        the platform refuses it unless ACL graph is off (``--enforce-eager``).
        """
        self._drain_launches(device)

    def _fence_decode(self, device: torch.device) -> None:
        """Drain the decode's launch, so a core exception names the decode.

        The mirror of :meth:`_fence_cache_write`, and needed for the same reason --
        with one twist that makes the decode worse to diagnose than the write. The
        first thing after a decode launch that synchronises with the device is the
        *next* step's staging copy in :meth:`_device_index`, so a decode that traps
        comes back as ``ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`` (507035) raised from
        ``copy_between_host_and_device_opapi`` -- an H2D copy of a handful of int32
        lengths, which neither reads nor writes anything a core could fault on. The
        copy is the victim of the queue, not the cause.

        Diagnostic only, and part of what ``VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS``
        costs: one synchronisation per layer per decode step.
        """
        self._drain_launches(device)

    def _drain_launches(self, device: torch.device) -> None:
        """Wait for everything queued so far. The primitive both fences are."""
        if device.type != "npu":
            return
        torch.npu.synchronize()

    def reshape_and_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ):
        if len(kv_cache) <= 1:
            return query, key, value, output

        if self.key_cache is None:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]
        if self.kv_sharing_target_layer_name is not None:
            if self.is_kv_producer:
                attn_metadata.reshape_cache_event.record()
            return query, key, value, output

        self._ensure_scale_cache(kv_cache)
        self._ensure_sink_cache(kv_cache, key.dtype)
        num_actual_tokens = attn_metadata.num_actual_tokens
        encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
        slots = attn_metadata.slot_mapping if encoder_decoder else attn_metadata.slot_mapping[:num_actual_tokens]

        # Packed, because the kernel indexes all three as raw global memory: a fused QKV
        # projection hands the backend column slices of one matrix, and a slot_mapping the
        # metadata builder sliced is a view. Materialised here rather than inside the call so
        # the validator sees the addresses the operator is actually handed.
        cached_key = (key if encoder_decoder else key[:num_actual_tokens]).contiguous()
        cached_value = (value if encoder_decoder else value[:num_actual_tokens]).contiguous()
        cached_slots = slots.to(torch.int32).contiguous()

        if envs_ascend.VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS:
            self._validate_cache_write(cached_key, cached_value, cached_slots)

        # The two decodes read different byte layouts, so the writer follows the decode.
        writer_name = (
            "npu_turboquant_cube_reshape_and_cache" if self.cube_decode else "npu_turboquant_reshape_and_cache"
        )
        write = getattr(torch.ops._C_ascend, writer_name)

        # Recorded here rather than after the launch because the launch is where the
        # process dies: the operands are described while they still exist, and the
        # record is on disk before the writer is asked to touch them.
        tracer = turboquant_tracer()
        if tracer.active:
            geometry = self._cache_geometry()
            # The ring and the case harvester first: both are host-side only, so they run
            # even where the full record below would be too expensive to leave on.
            tracer.observe(
                "reshape_and_cache",
                tensors={"key": cached_key, "value": cached_value, "slot_mapping": cached_slots},
                block_size=geometry["block_size"],
                num_tokens=int(cached_key.shape[0]),
                layer=self._trace_layer_name,
                step=self._trace_step,
                phase=self._trace_phase,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_size,
                num_blocks=geometry["num_blocks"],
            )
        if tracer.capture:
            tracer.record(
                "reshape_and_cache",
                **self._trace_caller_context(),
                operator=writer_name,
                launch_args={
                    "num_tokens": int(cached_key.shape[0]),
                    "num_scheduled_tokens": int(num_actual_tokens),
                    "num_kv_heads": self.num_kv_heads,
                    "head_dim": self.head_size,
                    **geometry,
                },
                slot_mapping=describe_indices(cached_slots, capacity=geometry["capacity"]),
                tensors=describe_tensors(
                    key=cached_key,
                    value=cached_value,
                    key_cache=self.key_cache,
                    value_cache=self.value_cache,
                    scale_cache=self.scale_cache,
                    slot_mapping=cached_slots,
                    pi_signs=self.pi_signs(key.device),
                    codec_tables=self.codec_tables(key.device, 1),
                ),
                bypassed=tracer.dry_run,
            )
        if not tracer.dry_run:
            write(
                cached_key,
                cached_value,
                self.key_cache,
                self.value_cache,
                self.scale_cache,
                cached_slots,
                self.pi_signs(key.device),
                self.codec_tables(key.device, 1),
            )
            if envs_ascend.VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS:
                self._fence_cache_write(key.device)
            # After the packed write, not instead of it: a sink is quantised into the paged
            # cache exactly as before, and kept here as well, so the decode's context stays
            # a contiguous prefix and only the merge below knows the difference.
            if self.sink_config.enabled:
                self._ingest_sinks(cached_key, cached_value, attn_metadata)
        # Announced either way: a dry run leaves the cache empty, but the rest of the
        # engine still has to be told the write path ran, or a KV connector waits for
        # an event that is never recorded and the run stops before it has traced anything.
        notify_kv_cache_written()
        return query, key, value, output

    def _decode_workspace(
        self, num_tokens: int, max_blocks_per_seq: int, block_size: int, device: torch.device
    ) -> torch.Tensor:
        """Return scratch for the decode's in-launch sequence reduction.

        The decode is one launch.  A sequence that fits the fused context limit
        writes its output directly and needs no scratch at all, so the operator
        asks for a workspace only when ``max_blocks_per_seq * block_size`` can
        exceed that limit: those sequences are split, and each split writes one
        partial per (token, head, split) that the same launch then reduces.
        That buffer used to be an ``at::empty``
        inside the operator, which put an allocation on every decode step and
        left the operator uncapturable: a graph replays the addresses it was
        captured with, and a fresh allocation each step is a different address.

        So the buffer lives here instead, is grown only when a decode needs more
        than any seen before, and is otherwise handed to the kernel unchanged.
        Steady-state decode allocates nothing.  "More" is deliberately not "a
        bigger batch": a long-context batch small enough to leave cores idle is
        split further along the sequence, so the peak requirement can sit at a
        *small* decode.

        The size comes from the operator's own arithmetic rather than a copy of
        it -- the split count depends on the device's vector core count, which
        only the C++ side knows -- and is memoised per shape so the hot path
        costs a dict lookup rather than an operator dispatch.
        """
        key = (num_tokens, max_blocks_per_seq)
        needed = self._workspace_floats.get(key)
        if needed is None:
            if len(self._workspace_floats) >= _WORKSPACE_MEMO_LIMIT:
                self._workspace_floats.clear()
            if self.cube_decode:
                # The Cube planner tiles by kv head, so it needs the kv head count too.
                needed = int(
                    torch.ops._C_ascend.npu_turboquant_cube_workspace_size(
                        num_tokens, self.num_heads, self.num_kv_heads, self.head_size, max_blocks_per_seq, block_size
                    )
                )
            else:
                needed = int(
                    torch.ops._C_ascend.npu_turboquant_workspace_size(
                        num_tokens, self.num_heads, self.head_size, max_blocks_per_seq, block_size
                    )
                )
            self._workspace_floats[key] = needed

        workspace = self.decode_workspace
        if workspace is not None and workspace.numel() >= needed:
            return workspace

        if _is_capturing():
            raise RuntimeError(
                "[vllm-ascend/turboquant] the decode workspace would have to grow to "
                f"{needed} float32 words during a graph capture "
                f"(num_tokens={num_tokens}, max_blocks_per_seq={max_blocks_per_seq}). "
                "Run this shape once outside capture so the buffer is sized first."
            )

        self.decode_workspace = torch.empty(needed, dtype=torch.float32, device=device)
        return self.decode_workspace

    def _decode_capacity(self) -> tuple[int, int]:
        """``(sequences, blocks per sequence)`` at the largest decode the config allows.

        The buffers below are grown on demand, and every growth is an allocation the
        step that needs it has to make. A graph capture cannot make one -- it replays
        the addresses it recorded -- so a buffer that first has to grow during a
        capture is a refusal (:func:`_is_capturing`), and the run stops at exactly the
        shape it was supposed to speed up.

        vLLM captures after one eager warmup per shape, so in practice the warmup
        sizes the buffer and the capture finds it big enough. "In practice" is the
        problem: it holds only while the warmup covers every shape the capture does,
        in the right order. Sizing the *first* allocation to the configured ceiling
        instead makes it hold unconditionally, for a buffer that is a few hundred
        kilobytes either way -- these hold one int32 per sequence and one block-table
        row per sequence, not activations.

        ``(0, 0)`` where the config cannot be read: the caller falls back to sizing by
        what the call in hand needs, which is the behaviour this replaces.
        """
        vllm_config = getattr(self, "vllm_config", None)
        if vllm_config is None:
            return 0, 0
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        model_config = getattr(vllm_config, "model_config", None)
        cache_config = getattr(vllm_config, "cache_config", None)
        limits = (
            getattr(scheduler_config, "max_num_seqs", None),
            getattr(model_config, "max_model_len", None),
            getattr(cache_config, "block_size", None),
        )
        # isinstance rather than truthiness: a partially built config -- or a test
        # double standing in for one -- answers with something that is not a number,
        # and sizing a buffer from it would be worse than not reserving at all.
        if not all(isinstance(limit, int) and limit > 0 for limit in limits):
            return 0, 0
        max_num_seqs, max_model_len, block_size = limits
        return max_num_seqs, cdiv(max_model_len, block_size)

    def _reserved_index_words(self, name: str) -> int:
        """How many int32 words to reserve for the ``name`` staging buffer up front.

        Only the two operands the decodes stage are known from the config -- one
        length per sequence, and one block-table row per sequence. Anything else
        is sized by the call that needs it.
        """
        max_num_seqs, max_blocks_per_seq = self._decode_capacity()
        if name == "seq_lens":
            return max_num_seqs
        if name == "block_tables":
            return max_num_seqs * max_blocks_per_seq
        return 0

    def _device_index(self, indices: torch.Tensor, device: torch.device, name: str) -> torch.Tensor:
        """Return ``indices`` as a contiguous int32 tensor in the kernels' global memory.

        ``AscendMetadata.seq_lens`` is deliberately a *host* tensor: the metadata
        builder takes it from ``seq_lens_cpu``, or copies it to the host if it has
        to, because the CANN paged attention the stock backend calls reads the
        lengths host-side to tile with. These kernels do not --
        ``contextLenGm_.GetValue(token)`` is a scalar read from global memory -- so
        handing them the metadata's tensor hands a vector core a host address. That
        is not a wrong answer, it is "the address for scalar to access GM is
        invalid" (error 264), reported from whichever later launch the runtime was
        working on.

        ``.to(torch.int32).contiguous()`` does not catch it: the tensor is already
        contiguous int32, so both calls return it unchanged, device and all.

        The copy lands in a persistent per-operand buffer rather than a fresh
        allocation for the reason the decode workspace does -- a graph replays the
        addresses it was captured with -- and an operand already on the device is
        returned untouched, so the path that was always correct costs nothing.

        The narrowing and the cast both happen on the operand's own device, before
        the transfer, so the copy is always same-dtype host-to-device and never asks
        the runtime to widen or narrow across the bus. The staging buffer is
        allocated in the operand's dtype and keyed by it, so the two cannot drift
        apart, and the invariant is checked rather than assumed -- a mismatch here
        would be a silent reinterpretation of the bytes the kernel then indexes with.

        Note for anyone reading a traceback that lands on the copy below: this is
        the first call after a decode launch that synchronises with the device, so
        it is where an *asynchronous* core fault from the previous launch surfaces.
        ``ACL_ERROR_RT_VECTOR_CORE_EXCEPTION`` (507035) raised from
        ``copy_between_host_and_device_opapi`` is a kernel that already trapped, not
        this transfer of a handful of int32 lengths: a copy touches no AI core.
        :meth:`_fence_decode` is what moves the report back onto the launch that
        caused it.
        """
        # The kernels index with int32 (``__gm__ int32_t *``); vLLM hands these over
        # as int64 on several paths. Cast first and in place on the source device, so
        # the transfer below moves int32 words to an int32 buffer.
        if indices.dtype != TURBOQUANT_INDEX_DTYPE:
            indices = indices.to(TURBOQUANT_INDEX_DTYPE)
        indices = indices.contiguous()
        if indices.device == device:
            return indices

        if self._device_index_buffers is None:
            self._device_index_buffers = {}
        needed = indices.numel()
        # Keyed by dtype as well as operand, so a buffer can never be reused under a
        # dtype it was not allocated for.
        key = (name, indices.dtype)
        buffer = self._device_index_buffers.get(key)
        if buffer is None or buffer.numel() < needed:
            if _is_capturing():
                raise RuntimeError(
                    f"[vllm-ascend/turboquant] {name} lives on {indices.device} and its staging buffer would "
                    f"have to grow to {needed} {indices.dtype} words during a graph capture. Run this shape "
                    "once outside capture so the buffer is sized first."
                )
            # Zeroed, not empty. Only the first ``needed`` words are ever copied into,
            # so everything past them is whatever the allocator handed back -- and a
            # reserved buffer is bigger than the step that allocated it needs, which
            # means there is now a tail that no step writes. An operand the kernels
            # index by a token counter must not be able to return uninitialised memory
            # for a token the host did not describe: zero is the only value here that
            # names a sequence of no length rather than an arbitrary one. Paid once,
            # at the single allocation this buffer ever gets.
            buffer = torch.zeros(max(needed, self._reserved_index_words(name)), dtype=indices.dtype, device=device)
            self._device_index_buffers[key] = buffer
        staged = buffer[:needed].view(indices.shape)
        if staged.dtype != indices.dtype:
            raise RuntimeError(
                f"[vllm-ascend/turboquant] the {name} staging buffer is {staged.dtype} but the operand is "
                f"{indices.dtype}; the copy would reinterpret the bytes the kernel indexes with"
            )
        staged.copy_(indices)
        return staged

    def _decode_seq_lens(self, attn_metadata: AscendMetadata, device: torch.device) -> torch.Tensor:
        """The context lengths, in the global memory the decode reads them from.

        ``AscendMetadata.seq_lens`` is a host tensor on purpose, so staging it was
        a host-to-device transfer *per layer per decode step* -- sixty of them a
        step on a sixty-layer model, each moving a few hundred int32 words, and
        each one pageable and therefore synchronous.

        ``seq_lens_device`` is the same batch's lengths left where the model runner
        already computes them: a persistent int32 device buffer, written on device
        and zero-padded past ``num_reqs``. Preferring it makes the whole thing free
        -- :meth:`_device_index` hands back an operand that is already int32,
        already contiguous and already on the right device, so nothing is copied
        and nothing is allocated. The zero padding is the right value too: a row
        the host did not describe reads as a sequence of no length.

        It is still routed through :meth:`_device_index` rather than used raw, so
        the dtype and residency invariants are checked rather than assumed, and so
        a metadata that predates the field still works by staging the host tensor.
        """
        lengths = getattr(attn_metadata, "seq_lens_device", None)
        # isinstance rather than "is not None", for the reason _decode_capacity uses
        # it: this is an optional field on a metadata object that several builders
        # and every test double construct, and staging something that is not a tensor
        # is a worse failure than falling back to the host tensor that always exists.
        if not isinstance(lengths, torch.Tensor):
            lengths = attn_metadata.seq_lens
        return self._device_index(lengths, device, "seq_lens")

    def _rotated_query(self, num_tokens: int, device: torch.device) -> torch.Tensor:
        """Return the fp32 buffer ``npu_turboquant_rotate_q`` writes into.

        The rotation used to happen inside the decode split, once per
        ``(token, kv_head, sequence split)`` and again per query head of the
        group.  It now happens once per ``(token, head)`` in its own operator,
        which needs somewhere to put the result.

        Persistent for exactly the reason ``decode_workspace`` is: a graph
        replays the addresses it was captured with, so a fresh allocation on
        every decode step would both cost an allocation and make the operator
        uncapturable.  The requirement is monotonic in ``num_tokens`` -- unlike
        the workspace's, which is not -- so growing it can only ever happen on
        the way up.
        """
        needed = num_tokens * self.num_heads * self.head_size
        buffer = self.rotated_query
        if buffer is not None and buffer.numel() >= needed:
            return buffer[:needed].view(num_tokens, self.num_heads, self.head_size)

        if _is_capturing():
            raise RuntimeError(
                "[vllm-ascend/turboquant] the rotated-query buffer would have to grow to "
                f"{needed} float32 words during a graph capture (num_tokens={num_tokens}). "
                "Run this shape once outside capture so the buffer is sized first."
            )

        # Sized to the largest decode the config allows, for the reason
        # _decode_capacity gives: a capture cannot be the step that grows it.
        max_num_seqs, _ = self._decode_capacity()
        reserved = max_num_seqs * self.num_heads * self.head_size
        # Zeroed for the reason the index staging buffers are: reserving past what the
        # allocating step needs leaves a tail no step writes, and a decode that reads a
        # row it was not handed should read zeros rather than an arbitrary float.
        self.rotated_query = torch.zeros(max(needed, reserved), dtype=torch.float32, device=device)
        return self.rotated_query[:needed].view(num_tokens, self.num_heads, self.head_size)

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        num_tokens = query.shape[0]
        # Hoisted above the first launch so the trace can describe the operands the
        # kernels are handed rather than the tensors they were derived from. All four
        # are host-side work -- a view, two dtype casts and a memoised buffer lookup --
        # so the live path is unchanged by the move.
        rotated_query = self._rotated_query(num_tokens, query.device)
        # Both are read from global memory by the kernel, and seq_lens reaches the
        # backend on the host. See _device_index.
        block_tables = self._device_index(attn_metadata.block_tables, query.device, "block_tables")
        seq_lens = self._decode_seq_lens(attn_metadata, query.device)
        workspace = self._decode_workspace(num_tokens, block_tables.shape[1], self.key_cache.shape[1], query.device)
        attention_output = output[:num_tokens].view(num_tokens, self.num_heads, self.head_size)

        tracer = turboquant_tracer()
        if tracer.active:
            self._trace_decode(
                tracer,
                operator="npu_turboquant_paged_attention",
                query=query,
                rotated_query=rotated_query,
                block_tables=block_tables,
                seq_lens=seq_lens,
                host_seq_lens=attn_metadata.seq_lens,
                workspace=workspace,
                attention_output=attention_output,
                output_gate=None,
                stage=None,
            )
            if tracer.dry_run:
                # Zero rather than left alone: `output` is whatever the runner last put
                # there, so an untouched buffer would hand the model stale activations
                # and make a dry run look like it computed something.
                attention_output.zero_()
                return output

        torch.ops._C_ascend.npu_turboquant_rotate_q(
            query.contiguous(),
            self.pi_signs(query.device),
            self.codec_tables(query.device, 1),
            self.hadamard16(query.device),
            rotated_query,
        )
        torch.ops._C_ascend.npu_turboquant_paged_attention(
            rotated_query,
            self.key_cache,
            self.value_cache,
            self.scale_cache,
            block_tables,
            seq_lens,
            self.codec_tables(query.device, TURBOQUANT_TILE_ROWS),
            workspace,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            attention_output,
        )
        if self.sink_config.enabled:
            self._fuse_decode_sinks(attention_output, query, rotated_query, attn_metadata)
        if not self.output_rotation_folded:
            self._unrotate_output(attention_output, rotated_query)
        if envs_ascend.VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS:
            self._fence_decode(query.device)
        return output

    def _trace_decode(
        self,
        tracer,
        *,
        operator: str,
        query: torch.Tensor,
        rotated_query: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        host_seq_lens: torch.Tensor,
        workspace: torch.Tensor,
        attention_output: torch.Tensor,
        output_gate: torch.Tensor | None,
        stage: TurboQuantOutputStage | None,
    ) -> None:
        """Record one decode launch, whichever of the two decodes is about to run.

        Both take the same cache, the same page geometry and the same reduction
        workspace, and differ only in whether the query rotation and the output
        stage are separate launches or part of this one -- so they are worth
        reading side by side in one file, under one schema.
        """
        geometry = self._cache_geometry()
        tracer.observe(
            "decode_cube" if self.cube_decode else "decode_aiv",
            tensors={
                "query": query,
                "block_tables": block_tables,
                "seq_lens": seq_lens,
                "workspace": workspace,
                "output": attention_output,
            },
            block_size=geometry["block_size"],
            num_tokens=int(query.shape[0]),
            # The metadata's own tensor, not the staged copy: seq_lens reaches the backend on
            # the host, and reading it there is free. _device_index has already moved the
            # operand the kernel reads onto the device, where a read would cost a sync.
            seq_lens=host_seq_lens,
            layer=self._trace_layer_name,
            step=self._trace_step,
            phase=self._trace_phase,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            num_blocks=geometry["num_blocks"],
            max_blocks_per_seq=int(block_tables.shape[1]),
            output_stage=None if stage is None else int(stage),
        )
        # Everything below builds its arguments eagerly, and describe_indices copies an index
        # tensor back from the device to do it. That is the price of the full record and must
        # not be charged to the ring, which exists to be cheap enough to leave on.
        if not tracer.capture:
            return
        tracer.record(
            "decode_attention",
            **self._trace_caller_context(),
            operator=operator,
            launch_args={
                "num_tokens": int(query.shape[0]),
                "num_heads": self.num_heads,
                "num_kv_heads": self.num_kv_heads,
                "head_dim": self.head_size,
                "scale": float(self.scale),
                "max_blocks_per_seq": int(block_tables.shape[1]),
                "workspace_floats": int(workspace.numel()),
                "output_stage": None if stage is None else int(stage),
                **geometry,
            },
            block_tables=describe_indices(block_tables, capacity=geometry["num_blocks"]),
            seq_lens=describe_indices(seq_lens, capacity=geometry["capacity"]),
            tensors=describe_tensors(
                query=query,
                rotated_query=rotated_query,
                key_cache=self.key_cache,
                value_cache=self.value_cache,
                scale_cache=self.scale_cache,
                block_tables=block_tables,
                seq_lens=seq_lens,
                workspace=workspace,
                output_gate=output_gate,
                output=attention_output,
            ),
            bypassed=tracer.dry_run,
        )

    def _cube_output_stage(self, output_gate: torch.Tensor | None) -> TurboQuantOutputStage:
        if self.output_rotation_folded and output_gate is not None:
            raise ValueError(
                "[vllm-ascend/turboquant] a layer with Pi folded into o_proj cannot carry an output gate: "
                "the gate sits between attention and o_proj, where the output is still rotated"
            )
        if self.sink_config.enabled:
            # The merge is a rescaling of the whole softmax, so it has to see the raw
            # rotated accumulator: un-rotating or gating first would have the launch apply
            # a transform to a numerator whose denominator is about to change. Both are
            # linear, so both are re-applied on the host after the merge, at the cost of
            # the launch fusion this stage exists to provide. Exposing the softmax
            # denominator as a decode output is what would give it back.
            return TurboQuantOutputStage.ROTATED_BASIS
        if self.output_rotation_folded:
            return TurboQuantOutputStage.ROTATED_BASIS
        return TurboQuantOutputStage.UNROTATED if output_gate is None else TurboQuantOutputStage.GATED

    def forward_cube_decode(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        output_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """One kv4fp8 Cube launch per step: rotate, attend, and write what o_proj reads.

        The launch is handed the raw query and rotates it into the persistent
        ``rotated_query`` buffer itself, ahead of its split tasks. Its output
        stage then leaves the result rotated for a folded layer, and otherwise
        un-rotates it in fp32 before the fp16 write -- and for a gated layer
        also multiplies it by ``sigmoid(output_gate)``. No other launch or copy
        follows.
        """
        num_tokens = query.shape[0]
        device = query.device
        stage = self._cube_output_stage(output_gate)
        gate = None if output_gate is None else output_gate[:num_tokens].contiguous()
        # Handed to the launch only where the launch is the one applying it; with sinks on
        # the stage is ROTATED_BASIS and the gate is applied below, after the merge.
        launch_gate = None if self.sink_config.enabled else gate
        # Staged onto the device for the reason forward_paged_attention stages them --
        # seq_lens reaches the backend on the host and this launch reads it from global
        # memory (see _device_index) -- and hoisted for the reason it hoists: the trace
        # has to name the tensors the launch is handed, not their sources.
        block_tables = self._device_index(attn_metadata.block_tables, device, "block_tables")
        seq_lens = self._decode_seq_lens(attn_metadata, device)
        workspace = self._decode_workspace(num_tokens, block_tables.shape[1], self.key_cache.shape[1], device)
        rotated_query = self._rotated_query(num_tokens, device)
        attention_output = output[:num_tokens].view(num_tokens, self.num_heads, self.head_size)

        tracer = turboquant_tracer()
        if tracer.active:
            self._trace_decode(
                tracer,
                operator="npu_turboquant_cube_decode",
                query=query,
                rotated_query=rotated_query,
                block_tables=block_tables,
                seq_lens=seq_lens,
                host_seq_lens=attn_metadata.seq_lens,
                workspace=workspace,
                attention_output=attention_output,
                output_gate=gate,
                stage=stage,
            )
            if tracer.dry_run:
                attention_output.zero_()
                return output

        torch.ops._C_ascend.npu_turboquant_cube_decode(
            query.contiguous(),
            launch_gate,
            self.pi_signs(device),
            self.codec_tables(device, 1),
            self.hadamard16(device),
            self.key_cache,
            self.value_cache,
            self.scale_cache,
            block_tables,
            seq_lens,
            workspace,
            rotated_query,
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            int(stage),
            attention_output,
        )
        if self.sink_config.enabled:
            self._fuse_decode_sinks(attention_output, query, rotated_query, attn_metadata)
            if not self.output_rotation_folded:
                self._unrotate_output(attention_output, rotated_query)
            if gate is not None:
                attention_output.mul_(torch.sigmoid(gate).view_as(attention_output))
        if envs_ascend.VLLM_ASCEND_TURBOQUANT_VALIDATE_SLOTS:
            self._fence_decode(device)
        return output

    def _rotate_value_for_prefill(self, value: torch.Tensor) -> torch.Tensor:
        """Return ``V~ = Pi V`` in ``value``'s own dtype.

        For a folded o_proj the prefill has to hand the projection the rotated
        basis the decode does. Rotating V is sufficient -- ``softmax(q k^T)(V Pi)
        = O Pi`` -- and it is ``H_KV`` vectors per token rather than the ``H_Q`` a
        rotation of the output would be. Q and K stay unrotated, so RoPE and the
        scores are untouched.

        Allocates: prefill is not replayed from a captured graph, and a buffer
        sized for the longest prompt would sit in HBM for the whole run.
        """
        value = value.contiguous()
        tracer = turboquant_tracer()
        if tracer.active:
            tracer.record(
                "prefill_rotate_value",
                **self._trace_caller_context(),
                operator="npu_turboquant_rotate_q",
                launch_args={"num_tokens": int(value.shape[0]), "num_kv_heads": self.num_kv_heads},
                tensors=describe_tensors(value=value, pi_signs=self.pi_signs(value.device)),
                bypassed=tracer.dry_run,
            )
            if tracer.dry_run:
                # The only TurboQuant launch a prefill makes. Handing the unrotated V
                # back leaves the dense attention below it running on a stock operator,
                # so a dry run still walks the whole prefill; it is the wrong basis for
                # a folded o_proj, which is no worse than the zeroed decode beside it.
                return value
        rotated = torch.empty(value.shape, dtype=torch.float32, device=value.device)
        torch.ops._C_ascend.npu_turboquant_rotate_q(
            value,
            self.pi_signs(value.device),
            self.codec_tables(value.device, 1),
            self.hadamard16(value.device),
            rotated,
        )
        return rotated.to(value.dtype)

    def _forward_prefill_no_cache(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Full-precision prefill over the current batch's K/V.

        Nothing is read back from the quantised cache here, so this is exact up
        to the rotation of V, which a folded o_proj undoes.
        """
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])
        # Packed for the reason reshape_and_cache packs: a fused QKV projection hands
        # the backend column slices of one matrix, and V never passes through RoPE, so
        # nothing on the way would pack it.  A prefill runs once per prompt, so the copy
        # is not on the hot path; the rotated V is already packed by its own operator.
        prefill_query = query[:num_tokens].contiguous()
        prefill_key = key[:num_tokens].contiguous()
        if self.output_rotation_folded:
            prefill_value = self._rotate_value_for_prefill(value[:num_tokens])
        else:
            prefill_value = value[:num_tokens].contiguous()
        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=prefill_query,
            key=prefill_key,
            value=prefill_value,
            atten_mask=attn_metadata.attn_mask,
            input_layout="TND",
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_qlen,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
            sparse_mode=3,
        )
        output[:num_tokens] = attn_output.view(num_tokens, self.num_heads, self.head_size)
        return output

    def forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor,
        output_gate: torch.Tensor | None = None,
    ):
        state = attn_metadata.attn_state
        if state == AscendAttentionState.DecodeOnly and self.cube_decode:
            return self.forward_cube_decode(query, attn_metadata, output, output_gate)
        if state == AscendAttentionState.DecodeOnly:
            output = self.forward_paged_attention(query, attn_metadata, output)
        elif state == AscendAttentionState.PrefillNoCache:
            output = self._forward_prefill_no_cache(query, key, value, attn_metadata, output)
        else:
            raise NotImplementedError(
                f"TurboQuant KV cache supports PrefillNoCache and DecodeOnly, got {state.name}. "
                "Chunked prefill needs a paged prefill kernel over the 4-bit cache."
            )
        if output_gate is not None:
            # Everything but the Cube decode hands o_proj's basis back ungated.
            num_tokens = query.shape[0]
            output[:num_tokens].mul_(torch.sigmoid(output_gate[:num_tokens]).view_as(output[:num_tokens]))
        return output

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, ...],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        output_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``output_gate``, from vllm::turboquant_gated_attention, is an attn_output_gate layer's raw gate,
        ``[num_tokens, num_heads, head_size]``: the output comes back multiplied by its sigmoid."""
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for AscendTurboQuantAttentionBackendImpl"
            )
        if self.sliding_window is not None:
            raise NotImplementedError("TurboQuant KV cache does not implement sliding window attention")

        num_tokens = query.shape[0]

        # Before the metadata check, not after: a step with no metadata is the profile
        # run that sizes the KV cache, and "this layer was entered with nothing to do"
        # is one of the things a trace is read to establish.
        tracer = turboquant_tracer()
        if tracer.active:
            self._trace_layer_name = getattr(layer, "layer_name", None) or getattr(layer, "prefix", None)
            self._trace_step += 1
            self._trace_phase = _attention_phase(attn_metadata)
            # Not `kv_cache or ()` and not `len(kv_cache)`: the profile run hands this
            # an empty tensor rather than the tuple of planes a served step does, and a
            # tensor answers neither question -- it raises, which would make the trace
            # the crash it exists to explain. See turboquant_trace.cache_planes.
            planes = cache_planes(kv_cache)
            tracer.record(
                "forward",
                **self._trace_caller_context(),
                launch_args={
                    "num_tokens": num_tokens,
                    "num_actual_tokens": getattr(attn_metadata, "num_actual_tokens", None),
                    "num_heads": self.num_heads,
                    "num_kv_heads": self.num_kv_heads,
                    "head_dim": self.head_size,
                    "kv_cache_planes": len(planes),
                },
                # The capacity is None until the cache is bound, three lines below: the
                # very first forward of a layer is exactly the one that has not bound it.
                slot_mapping=describe_indices(
                    getattr(attn_metadata, "slot_mapping", None), capacity=self._cache_geometry()["capacity"]
                ),
                tensors=describe_tensors(
                    query=query,
                    key=key,
                    value=value,
                    output=output,
                    output_gate=output_gate,
                    **{f"kv_cache[{i}]": plane for i, plane in enumerate(planes)},
                ),
            )

        if attn_metadata is None:
            return output.fill_(0)

        if self.key_cache is None and kv_cache is not None and len(kv_cache) >= 2:
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        if key is not None and value is not None:
            query, key, value, output = self.reshape_and_cache(query, key, value, kv_cache, attn_metadata, output)

        attn_output = self.forward_impl(query, key, value, kv_cache, attn_metadata, output, output_gate)
        output[:num_tokens] = attn_output[:num_tokens]
        return output


def activate_turboquant_backend(layer: torch.nn.Module) -> None:
    """Swap an already-built attention layer onto the TurboQuant impl.

    Mirrors how ``AscendC8KVCacheAttentionMethod`` upgrades a layer, so a
    quantization scheme can opt a model in without forking the model runner.
    """
    layer.kv_cache_torch_dtype = torch.int8
    impl = getattr(layer, "impl", None)
    if impl is not None:
        impl.__class__ = AscendTurboQuantAttentionBackendImpl
        impl.scale_cache = None
        impl._pi_signs = None
        impl.decode_workspace = None
        impl._workspace_floats = {}
        impl.rotated_query = None
        impl._hadamard16 = None
        impl._device_index_buffers = None
        impl.sink_config = TurboQuantSinkConfig.from_env()
        impl.sink_cache = None
        impl._trace_step = 0
        impl._trace_phase = "unknown"
        impl.cube_decode = turboquant_cube_decode_selected(_model_dtype(impl))
        vllm_config = getattr(impl, "vllm_config", None)
        hf_config = None if vllm_config is None else vllm_config.model_config.hf_config
        layer_name = getattr(layer, "layer_name", "")
        impl._trace_layer_name = layer_name or None
        impl.output_rotation_folded = output_rotation_is_folded(hf_config, layer_name, impl.head_size)
        logger.debug(
            "[vllm-ascend/turboquant] %s: %s decode; o_proj %s",
            layer_name,
            "kv4fp8 Cube" if impl.cube_decode else "AIV",
            "is Pi-folded; the decode output stays rotated"
            if impl.output_rotation_folded
            else "is not folded; the decode un-rotates its output on the device",
        )
