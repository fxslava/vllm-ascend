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
"""

import enum

import torch
import torch_npu
from vllm.logger import logger
from vllm.v1.attention.backend import AttentionLayer, AttentionType  # type: ignore

import vllm_ascend.envs as envs_ascend
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionState,
    AscendMetadata,
)
from vllm_ascend.attention.turboquant_rotation import output_rotation_is_folded, turboquant_pi_signs
from vllm_ascend.attention.utils import notify_kv_cache_written

TURBOQUANT_PACK_FACTOR = 2

TURBOQUANT_BURST_FLOATS = 8

TURBOQUANT_TILE_ROWS = 16

TURBOQUANT_LEVELS = 16

TURBOQUANT_LLOYD_MAX_CENTROIDS = (
    -2.7325895709951710, -2.0690172265313920, -1.6180463860218863, -1.2562311973471796,
    -0.9423404564869651, -0.6567591185324659, -0.3880482994902919, -0.1283950298511473,
    0.1283950298511473, 0.3880482994902919, 0.6567591185324659, 0.9423404564869651,
    1.2562311973471796, 1.6180463860218863, 2.0690172265313920, 2.7325895709951710,
)

TURBOQUANT_LLOYD_MAX_THRESHOLDS = (
    -2.4008033987632817, -1.8435318062766393, -1.4371387916845330, -1.0992858269170722,
    -0.7995497875097155, -0.5224037090113789, -0.2582216646707196, 0.0,
    0.2582216646707196, 0.5224037090113789, 0.7995497875097155, 1.0992858269170722,
    1.4371387916845330, 1.8435318062766393, 2.4008033987632817,
)

_CODEC_TABLE_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}
_HADAMARD16_CACHE: dict[str, torch.Tensor] = {}

TURBOQUANT_ROTATE_TILE = 16

# The Cube decode reads a block one 64-row tile at a time.
TURBOQUANT_CUBE_TILE_ROWS = 64

_WORKSPACE_MEMO_LIMIT = 1024


class TurboQuantOutputStage(enum.IntEnum):
    """What the Cube decode writes; mirrors ``TurboQuantOutputStage`` in turboquant_layout.h."""

    # softmax(q k^T) Pi v, for an o_proj with Pi folded in.
    ROTATED_BASIS = 0
    # Pi applied once more to the fp32 accumulator before the fp16 cast.
    UNROTATED = 1
    # UNROTATED, then multiplied by sigmoid(gate): an attn_output_gate layer.
    GATED = 2


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
    """Whether an ACL graph capture is in progress.

    Reading the flag needs a live forward context, and the answer is only ever
    used to refuse an allocation that would be unsafe *inside* a capture. Where
    there is no forward context at all there is no capture either, so a missing
    one is answered rather than raised -- that keeps a diagnostic from turning
    into the failure it exists to describe.
    """
    try:
        return bool(_EXTRA_CTX.capturing)
    except (AssertionError, AttributeError, RuntimeError):
        return False


def turboquant_hadamard16(device: torch.device) -> torch.Tensor:
    """Return the 16x16 fp16 Hadamard constant the Cube rotation path uses.

    ``H[i][j] = (-1) ** popcount(i & j)``: Sylvester's construction, symmetric,
    holding only +-1 -- both exactly representable in fp16, so the Cube stage of
    the rotation contributes no arithmetic error of its own.

    Sylvester also factorises the transform. For ``D = 16 R`` the butterfly
    stages of stride 1, 2, 4 and 8 are exactly a right multiply by this matrix,
    whatever ``D`` is, so four stages of the query's rotation collapse into a
    single Mmad and only the ``log2(R)`` whole-row stages above stride 8 stay on
    the vector unit.  Mirrors ``FillHadamard16Half`` in
    ``csrc/attention/turboquant/op_host/turboquant_tiling.cpp``.
    """
    key = str(device)
    cached = _HADAMARD16_CACHE.get(key)
    if cached is not None:
        return cached
    rows = []
    for i in range(TURBOQUANT_ROTATE_TILE):
        rows.append([
            -1.0 if bin(i & j).count("1") & 1 else 1.0
            for j in range(TURBOQUANT_ROTATE_TILE)
        ])
    matrix = torch.tensor(rows, dtype=torch.float16, device=device)
    _HADAMARD16_CACHE[key] = matrix
    return matrix


def turboquant_scale_slot(num_kv_heads: int) -> int:
    """fp32 lanes one token occupies in the scale plane.

    A token's K scales for every kv head are followed by its V scales, padded to
    a whole 32-byte burst.  That padding is what lets the kernel scatter a
    token's scales with one aligned ``DataCopy`` instead of ``2 * num_kv_heads``
    four-byte writes, which were both a DMA-transaction disaster and a hazard --
    two cores could land in the same 32-byte line.  It costs 25% on top of the
    payload at ``num_kv_heads == 2`` and 12.5% from 4 upward.
    """
    lanes = 2 * num_kv_heads
    return -(-lanes // TURBOQUANT_BURST_FLOATS) * TURBOQUANT_BURST_FLOATS


def turboquant_codec_tables(head_size: int, batch_rows: int, device: torch.device) -> torch.Tensor:
    """Build the codec's constant-table image.

    The sign patterns, XOR shuffle offsets and nibble split tables are a pure
    function of ``(head_size, batch_rows)``.  They used to be regenerated inside
    the kernel on every launch -- roughly forty dependent vector instructions and
    their pipe barriers on the critical path of each decode step.  The host
    builds them once here and the kernel pulls the finished image into UB with a
    single aligned ``DataCopy``.

    Layout, in 4-byte words, mirroring ``TurboQuantCodec<4>::ConstTableWords``:

        [0, 6D)         sign_[s] then xorOffset_[s], for strides 1, 2, 4
        [6D, 7D)        evenOffset_ then oddOffset_, D/2 words each
        [7D, 7D + B)    expandOffset_
        [7D + B, +B)    oddSelect_                      (B = D * batch_rows)
        [7D + 2B, +16) centroid_, the Lloyd-Max reconstruction levels

    ``sign_``, ``oddSelect_`` and ``centroid_`` are fp32 bit patterns, the rest
    uint32 byte offsets for Gather; everything is four bytes wide so one int32
    copy moves the lot and the device needs no cast.  The centroid table is
    appended rather than prepended so that adding it left every pre-existing
    offset in the image unchanged; ``head_size`` is a power of two at least 64,
    so every section boundary is still a whole 32-byte burst.
    """
    key = (head_size, batch_rows, str(device))
    cached = _CODEC_TABLE_CACHE.get(key)
    if cached is not None:
        return cached

    word = 4
    channels = torch.arange(head_size, dtype=torch.int64)
    parts: list[torch.Tensor] = []
    for stage in range(3):
        stride = 1 << stage
        sign = (1 - 2 * ((channels // stride) & 1)).to(torch.float32)
        parts.append(sign.view(torch.int32))
        parts.append((word * (channels ^ stride)).to(torch.int32))

    pairs = torch.arange(head_size // TURBOQUANT_PACK_FACTOR, dtype=torch.int64)
    parts.append((word * TURBOQUANT_PACK_FACTOR * pairs).to(torch.int32))
    parts.append((word * TURBOQUANT_PACK_FACTOR * pairs + word).to(torch.int32))

    batch = torch.arange(head_size * batch_rows, dtype=torch.int64)
    parts.append((word * (batch // TURBOQUANT_PACK_FACTOR)).to(torch.int32))
    parts.append((batch % TURBOQUANT_PACK_FACTOR).to(torch.float32).view(torch.int32))

    centroids = torch.tensor(TURBOQUANT_LLOYD_MAX_CENTROIDS, dtype=torch.float32)
    parts.append(centroids.view(torch.int32))

    tables = torch.cat(parts).contiguous().to(device)
    expected = 7 * head_size + 2 * head_size * batch_rows + TURBOQUANT_LEVELS
    if tables.numel() != expected:
        raise RuntimeError(f"codec table image is {tables.numel()} words, expected {expected}")
    _CODEC_TABLE_CACHE[key] = tables
    return tables


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

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.scale_cache: torch.Tensor | None = None
        self._pi_signs: torch.Tensor | None = None
        self.decode_workspace: torch.Tensor | None = None
        self._workspace_floats: dict[tuple[int, int], int] = {}
        self.rotated_query: torch.Tensor | None = None
        self._hadamard16: torch.Tensor | None = None
        self.output_rotation_folded = False
        self.cube_decode = turboquant_cube_decode_selected(_model_dtype(self))

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
                raise RuntimeError(
                    f"[vllm-ascend/turboquant] the scale plane must be float32, got {provided.dtype}"
                )
            self.scale_cache = provided
            return

        if _is_capturing():
            raise RuntimeError(
                f"[vllm-ascend/turboquant] the {shape} scale plane would have to be allocated during a "
                "graph capture. Run this layer once outside capture, or let the model runner carve the "
                "plane from the KV cache allocation so there is nothing left to allocate here."
            )
        self.scale_cache = torch.zeros(shape, dtype=torch.float32, device=kv_cache[0].device)

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
        num_actual_tokens = attn_metadata.num_actual_tokens
        encoder_decoder = self.attn_type == AttentionType.ENCODER_DECODER
        slots = attn_metadata.slot_mapping if encoder_decoder else attn_metadata.slot_mapping[:num_actual_tokens]

        cached_key = key if encoder_decoder else key[:num_actual_tokens]
        cached_value = value if encoder_decoder else value[:num_actual_tokens]

        # The two decodes read different byte layouts, so the writer follows the decode.
        write = (
            torch.ops._C_ascend.npu_turboquant_cube_reshape_and_cache
            if self.cube_decode
            else torch.ops._C_ascend.npu_turboquant_reshape_and_cache
        )
        write(
            cached_key.contiguous(),
            cached_value.contiguous(),
            self.key_cache,
            self.value_cache,
            self.scale_cache,
            slots.to(torch.int32).contiguous(),
            self.pi_signs(key.device),
            self.codec_tables(key.device, 1),
        )
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

        self.rotated_query = torch.empty(needed, dtype=torch.float32, device=device)
        return self.rotated_query.view(num_tokens, self.num_heads, self.head_size)

    def forward_paged_attention(
        self,
        query: torch.Tensor,
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        num_tokens = query.shape[0]
        rotated_query = self._rotated_query(num_tokens, query.device)
        torch.ops._C_ascend.npu_turboquant_rotate_q(
            query.contiguous(),
            self.pi_signs(query.device),
            self.codec_tables(query.device, 1),
            self.hadamard16(query.device),
            rotated_query,
        )
        block_tables = attn_metadata.block_tables.to(torch.int32).contiguous()
        attention_output = output[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
        torch.ops._C_ascend.npu_turboquant_paged_attention(
            rotated_query,
            self.key_cache,
            self.value_cache,
            self.scale_cache,
            block_tables,
            attn_metadata.seq_lens.to(torch.int32).contiguous(),
            self.codec_tables(query.device, TURBOQUANT_TILE_ROWS),
            self._decode_workspace(num_tokens, block_tables.shape[1], self.key_cache.shape[1], query.device),
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            attention_output,
        )
        if not self.output_rotation_folded:
            torch.ops._C_ascend.npu_turboquant_rotate_q(
                attention_output,
                self.pi_signs(query.device),
                self.codec_tables(query.device, 1),
                self.hadamard16(query.device),
                rotated_query,
            )
            attention_output.copy_(rotated_query)
        return output

    def _cube_output_stage(self, output_gate: torch.Tensor | None) -> TurboQuantOutputStage:
        if self.output_rotation_folded:
            if output_gate is not None:
                raise ValueError(
                    "[vllm-ascend/turboquant] a layer with Pi folded into o_proj cannot carry an output gate: "
                    "the gate sits between attention and o_proj, where the output is still rotated"
                )
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
        block_tables = attn_metadata.block_tables.to(torch.int32).contiguous()
        attention_output = output[:num_tokens].view(num_tokens, self.num_heads, self.head_size)
        torch.ops._C_ascend.npu_turboquant_cube_decode(
            query.contiguous(),
            gate,
            self.pi_signs(device),
            self.codec_tables(device, 1),
            self.hadamard16(device),
            self.key_cache,
            self.value_cache,
            self.scale_cache,
            block_tables,
            attn_metadata.seq_lens.to(torch.int32).contiguous(),
            self._decode_workspace(num_tokens, block_tables.shape[1], self.key_cache.shape[1], device),
            self._rotated_query(num_tokens, device),
            self.num_kv_heads,
            self.num_heads,
            self.scale,
            int(stage),
            attention_output,
        )
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
        impl.cube_decode = turboquant_cube_decode_selected(_model_dtype(impl))
        vllm_config = getattr(impl, "vllm_config", None)
        hf_config = None if vllm_config is None else vllm_config.model_config.hf_config
        layer_name = getattr(layer, "layer_name", "")
        impl.output_rotation_folded = output_rotation_is_folded(hf_config, layer_name, impl.head_size)
        logger.debug(
            "[vllm-ascend/turboquant] %s: %s decode; o_proj %s",
            layer_name,
            "kv4fp8 Cube" if impl.cube_decode else "AIV",
            "is Pi-folded; the decode output stays rotated"
            if impl.output_rotation_folded
            else "is not folded; the decode un-rotates its output on the device",
        )
